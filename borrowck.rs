//! Borrow checker pass for Jules.
//!
//! This pass is intentionally conservative and lexical:
//! - enforces `&mut` exclusivity
//! - rejects writes while borrowed
//! - tracks simple moves (`let y = x`, `y = x`) and use-after-move

use std::collections::HashMap;

use crate::ast::{AssignOpKind, Block, Expr, Item, Pattern, Program, Stmt, UnOpKind};
use crate::lexer::Span;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub span: Span,
    pub message: String,
    pub labels: Vec<(Span, String)>,
}

#[derive(Debug, Default, Clone)]
pub struct Diagnostics {
    pub items: Vec<Diagnostic>,
}

impl Diagnostics {
    fn error(&mut self, span: Span, message: impl Into<String>) {
        self.items.push(Diagnostic {
            severity: Severity::Error,
            span,
            message: message.into(),
            labels: vec![],
        });
    }
}

#[derive(Debug, Default, Clone)]
struct LoanState {
    imm: usize,
    mut_: usize,
}

#[derive(Debug, Default, Clone)]
struct VarState {
    moved: bool,
}

#[derive(Debug)]
struct BorrowChecker {
    diag: Diagnostics,
    loans_by_target: HashMap<String, LoanState>,
    ref_binding_to_target: HashMap<String, (String, bool)>, // ref_var -> (target, mut?)
    var_state: HashMap<String, VarState>,
    scope_bindings: Vec<Vec<String>>, // variable names declared in lexical scopes
}

impl Default for BorrowChecker {
    fn default() -> Self {
        Self {
            diag: Diagnostics::default(),
            loans_by_target: HashMap::new(),
            ref_binding_to_target: HashMap::new(),
            var_state: HashMap::new(),
            scope_bindings: vec![],
        }
    }
}

impl BorrowChecker {
    fn push_scope(&mut self) {
        self.scope_bindings.push(vec![]);
    }

    fn pop_scope(&mut self) {
        if let Some(vars) = self.scope_bindings.pop() {
            for v in vars {
                // Dropping a reference variable ends its loan.
                self.release_ref_binding(&v);
                self.var_state.remove(&v);
            }
        }
    }

    fn bind_var(&mut self, name: &str) {
        self.var_state.entry(name.to_string()).or_default().moved = false;
        if let Some(scope) = self.scope_bindings.last_mut() {
            scope.push(name.to_string());
        }
    }

    fn mark_moved(&mut self, name: &str) {
        if let Some(st) = self.var_state.get_mut(name) {
            st.moved = true;
        }
    }

    fn mark_initialized(&mut self, name: &str) {
        self.var_state.entry(name.to_string()).or_default().moved = false;
    }

    fn release_ref_binding(&mut self, ref_name: &str) {
        let Some((target, is_mut)) = self.ref_binding_to_target.remove(ref_name) else {
            return;
        };
        let mut remove_target = false;
        if let Some(st) = self.loans_by_target.get_mut(&target) {
            if is_mut {
                st.mut_ = st.mut_.saturating_sub(1);
            } else {
                st.imm = st.imm.saturating_sub(1);
            }
            remove_target = st.imm == 0 && st.mut_ == 0;
        }
        if remove_target {
            self.loans_by_target.remove(&target);
        }
    }

    fn try_borrow(&mut self, target: &str, is_mut: bool, span: Span) -> bool {
        if self.var_state.get(target).map(|v| v.moved).unwrap_or(false) {
            self.diag.error(
                span,
                format!("cannot borrow `{target}` because it has been moved"),
            );
            return false;
        }
        let st = self.loans_by_target.entry(target.to_string()).or_default();
        if is_mut {
            if st.mut_ > 0 || st.imm > 0 {
                self.diag.error(
                    span,
                    format!("cannot mutably borrow `{target}` because it is already borrowed"),
                );
                return false;
            }
            st.mut_ += 1;
            true
        } else {
            if st.mut_ > 0 {
                self.diag.error(
                    span,
                    format!(
                        "cannot immutably borrow `{target}` because it is already mutably borrowed"
                    ),
                );
                return false;
            }
            st.imm += 1;
            true
        }
    }

    fn check_read(&mut self, name: &str, span: Span) {
        if self.var_state.get(name).map(|v| v.moved).unwrap_or(false) {
            self.diag
                .error(span, format!("use of moved value `{name}`"));
        }
        if let Some(st) = self.loans_by_target.get(name) {
            if st.mut_ > 0 {
                self.diag.error(
                    span,
                    format!("cannot use `{name}` while it is mutably borrowed"),
                );
            }
        }
    }

    fn check_write(&mut self, name: &str, span: Span) {
        if let Some(st) = self.loans_by_target.get(name) {
            if st.mut_ > 0 || st.imm > 0 {
                self.diag
                    .error(span, format!("cannot assign to `{name}` while it is borrowed"));
            }
        }
    }

    fn move_ident(&mut self, name: &str, span: Span) {
        if let Some(st) = self.loans_by_target.get(name) {
            if st.mut_ > 0 || st.imm > 0 {
                self.diag.error(
                    span,
                    format!("cannot move `{name}` while it is borrowed"),
                );
                return;
            }
        }
        self.check_read(name, span);
        self.mark_moved(name);
    }

    fn bind_reference_var(&mut self, ref_var: &str, target: &str, is_mut: bool) {
        self.release_ref_binding(ref_var);
        self.ref_binding_to_target
            .insert(ref_var.to_string(), (target.to_string(), is_mut));
    }

    fn expr_as_simple_borrow(expr: &Expr) -> Option<(&str, bool, Span)> {
        if let Expr::UnOp { op, expr, span } = expr {
            if let Expr::Ident { name, .. } = &**expr {
                return match op {
                    UnOpKind::Ref => Some((name, false, *span)),
                    UnOpKind::RefMut => Some((name, true, *span)),
                    _ => None,
                };
            }
        }
        None
    }

    fn expr_as_simple_move(expr: &Expr) -> Option<(&str, Span)> {
        if let Expr::Ident { name, span } = expr {
            Some((name, *span))
        } else {
            None
        }
    }

    fn bind_pattern_vars(&mut self, pat: &Pattern) {
        match pat {
            Pattern::Ident { name, .. } => self.bind_var(name),
            Pattern::Tuple { elems, .. } => {
                for p in elems {
                    self.bind_pattern_vars(p);
                }
            }
            Pattern::Struct { fields, .. } => {
                for (_, p) in fields {
                    if let Some(p) = p {
                        self.bind_pattern_vars(p);
                    }
                }
            }
            Pattern::Enum { inner, .. } => {
                for p in inner {
                    self.bind_pattern_vars(p);
                }
            }
            Pattern::Or { arms, .. } => {
                // Conservative: bind all names from all arms.
                for p in arms {
                    self.bind_pattern_vars(p);
                }
            }
            _ => {}
        }
    }

    fn first_ident_in_pattern<'a>(pat: &'a Pattern) -> Option<&'a str> {
        match pat {
            Pattern::Ident { name, .. } => Some(name),
            Pattern::Tuple { elems, .. } => elems.iter().find_map(Self::first_ident_in_pattern),
            Pattern::Struct { fields, .. } => fields
                .iter()
                .find_map(|(_, p)| p.as_ref().and_then(Self::first_ident_in_pattern)),
            Pattern::Enum { inner, .. } => inner.iter().find_map(Self::first_ident_in_pattern),
            Pattern::Or { arms, .. } => arms.iter().find_map(Self::first_ident_in_pattern),
            _ => None,
        }
    }

    fn check_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Ident { name, span } => self.check_read(name, *span),
            Expr::UnOp { op, expr, span } => {
                if let Expr::Ident { name, .. } = &**expr {
                    match op {
                        UnOpKind::Ref => {
                            let _ = self.try_borrow(name, false, *span);
                        }
                        UnOpKind::RefMut => {
                            let _ = self.try_borrow(name, true, *span);
                        }
                        _ => {}
                    }
                }
                self.check_expr(expr);
            }
            Expr::Assign { op, target, value, span } => {
                match &**target {
                    Expr::Ident { name, .. } => {
                        self.check_write(name, *span);
                        if matches!(op, AssignOpKind::Assign) {
                            self.mark_initialized(name);
                        } else {
                            self.check_read(name, *span);
                        }
                        // Rebinding a reference variable drops the old borrow.
                        self.release_ref_binding(name);
                        if let Some((src, mv_span)) = Self::expr_as_simple_move(value) {
                            self.move_ident(src, mv_span);
                        }
                        if let Some((borrowed, is_mut, bspan)) = Self::expr_as_simple_borrow(value) {
                            if self.try_borrow(borrowed, is_mut, bspan) {
                                self.bind_reference_var(name, borrowed, is_mut);
                            }
                        }
                    }
                    Expr::UnOp { op: UnOpKind::Deref, expr: inner, .. } => {
                        if let Expr::Ident { name: ref_name, span: ref_span } = &**inner {
                            self.check_read(ref_name, *ref_span);
                            match self.ref_binding_to_target.get(ref_name) {
                                Some((_, true)) => {}
                                Some((target, false)) => self.diag.error(
                                    *span,
                                    format!(
                                        "cannot assign through immutable reference `{ref_name}` to `{target}`"
                                    ),
                                ),
                                None => self.diag.error(
                                    *span,
                                    format!("cannot dereference non-reference `{ref_name}` for assignment"),
                                ),
                            }
                        } else {
                            self.diag.error(*span, "unsupported assignment target through dereference");
                        }
                    }
                    _ => self.check_expr(target),
                }
                self.check_expr(value);
            }
            Expr::BinOp { lhs, rhs, .. }
            | Expr::MatMul { lhs, rhs, .. }
            | Expr::HadamardMul { lhs, rhs, .. }
            | Expr::HadamardDiv { lhs, rhs, .. }
            | Expr::TensorConcat { lhs, rhs, .. }
            | Expr::KronProd { lhs, rhs, .. }
            | Expr::OuterProd { lhs, rhs, .. }
            | Expr::Pow { base: lhs, exp: rhs, .. } => {
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            Expr::Field { object, .. } | Expr::Grad { inner: object, .. } => self.check_expr(object),
            Expr::Index { object, indices, .. } => {
                self.check_expr(object);
                for i in indices {
                    self.check_expr(i);
                }
            }
            Expr::Call { func, args, named, .. } => {
                self.check_expr(func);
                for a in args {
                    self.check_expr(a);
                    if let Some((name, sp)) = Self::expr_as_simple_move(a) {
                        self.move_ident(name, sp);
                    }
                }
                for (_, a) in named {
                    self.check_expr(a);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.check_expr(receiver);
                for a in args {
                    self.check_expr(a);
                    if let Some((name, sp)) = Self::expr_as_simple_move(a) {
                        self.move_ident(name, sp);
                    }
                }
            }
            Expr::Range { lo, hi, .. } => {
                if let Some(lo) = lo {
                    self.check_expr(lo);
                }
                if let Some(hi) = hi {
                    self.check_expr(hi);
                }
            }
            Expr::Cast { expr, .. } => self.check_expr(expr),
            Expr::IfExpr { cond, then, else_, .. } => {
                self.check_expr(cond);
                self.check_block(then);
                if let Some(else_) = else_ {
                    self.check_block(else_);
                }
            }
            Expr::Closure { body, .. } => self.check_expr(body),
            Expr::Block(b) => self.check_block(b),
            Expr::Tuple { elems, .. } | Expr::ArrayLit { elems, .. } => {
                for e in elems {
                    self.check_expr(e);
                }
            }
            Expr::StructLit { fields, .. } => {
                for (_, e) in fields {
                    self.check_expr(e);
                }
            }
            Expr::VecCtor { elems, .. } => {
                for e in elems {
                    self.check_expr(e);
                }
            }
            Expr::IntLit { .. }
            | Expr::FloatLit { .. }
            | Expr::BoolLit { .. }
            | Expr::StrLit { .. }
            | Expr::Path { .. } => {}
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { pattern, init, .. } => {
                if let Some(init) = init {
                    self.check_expr(init);
                }
                self.bind_pattern_vars(pattern);
                if let Some(binding_name) = Self::first_ident_in_pattern(pattern) {
                    if let Some(init) = init {
                        if let Some((src, span)) = Self::expr_as_simple_move(init) {
                            self.move_ident(src, span);
                        }
                        if let Some((borrowed, is_mut, bspan)) = Self::expr_as_simple_borrow(init) {
                            if self.try_borrow(borrowed, is_mut, bspan) {
                                self.bind_reference_var(binding_name, borrowed, is_mut);
                            }
                        }
                    }
                    self.mark_initialized(binding_name);
                }
            }
            Stmt::Expr { expr, .. } => self.check_expr(expr),
            Stmt::Return { value, .. } | Stmt::Break { value, .. } => {
                if let Some(v) = value {
                    self.check_expr(v);
                    if let Some((name, span)) = Self::expr_as_simple_move(v) {
                        self.move_ident(name, span);
                    }
                }
            }
            Stmt::Continue { .. } => {}
            Stmt::ForIn { pattern, iter, body, .. } => {
                self.check_expr(iter);
                self.push_scope();
                self.bind_pattern_vars(pattern);
                self.check_block(body);
                self.pop_scope();
            }
            Stmt::EntityFor { body, .. }
            | Stmt::While { body, .. }
            | Stmt::Loop { body, .. } => self.check_block(body),
            Stmt::If { cond, then, else_, .. } => {
                self.check_expr(cond);
                self.check_block(then);
                if let Some(else_) = else_ {
                    match &**else_ {
                        crate::ast::IfOrBlock::If(s) => self.check_stmt(s),
                        crate::ast::IfOrBlock::Block(b) => self.check_block(b),
                    }
                }
            }
            Stmt::Match { expr, arms, .. } => {
                self.check_expr(expr);
                for arm in arms {
                    if let Some(g) = &arm.guard {
                        self.check_expr(g);
                    }
                    self.check_expr(&arm.body);
                }
            }
            Stmt::Item(_) => {}
            Stmt::ParallelFor(pf) => {
                self.check_expr(&pf.iter);
                self.check_block(&pf.body);
            }
            Stmt::Spawn(b) => self.check_block(&b.body),
            Stmt::Sync(b) => self.check_block(&b.body),
            Stmt::Atomic(b) => self.check_block(&b.body),
        }
    }

    fn check_block(&mut self, block: &Block) {
        self.push_scope();
        for s in &block.stmts {
            self.check_stmt(s);
        }
        if let Some(tail) = &block.tail {
            self.check_expr(tail);
            if let Some((name, span)) = Self::expr_as_simple_move(tail) {
                self.move_ident(name, span);
            }
        }
        self.pop_scope();
    }
}

pub fn jules_borrowck(program: &Program) -> Diagnostics {
    let mut ck = BorrowChecker::default();
    ck.push_scope();
    for item in &program.items {
        match item {
            Item::Fn(f) => {
                if let Some(body) = &f.body {
                    ck.push_scope();
                    for p in &f.params {
                        ck.bind_var(&p.name);
                    }
                    ck.check_block(body);
                    ck.pop_scope();
                }
            }
            Item::System(s) => {
                ck.push_scope();
                for p in &s.params {
                    ck.bind_var(&p.name);
                }
                ck.check_block(&s.body);
                ck.pop_scope();
            }
            Item::Mod { items: Some(items), .. } => {
                for it in items {
                    if let Item::Fn(f) = it {
                        if let Some(body) = &f.body {
                            ck.push_scope();
                            for p in &f.params {
                                ck.bind_var(&p.name);
                            }
                            ck.check_block(body);
                            ck.pop_scope();
                        }
                    }
                }
            }
            _ => {}
        }
    }
    ck.pop_scope();
    ck.diag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;

    fn borrow_errors(source: &str) -> usize {
        let mut lx = Lexer::new(source);
        let (tokens, lex_errs) = lx.tokenize();
        assert!(lex_errs.is_empty(), "lexer errors: {lex_errs:?}");
        let mut p = Parser::new(tokens);
        let prog = p.parse_program();
        assert!(p.errors.is_empty(), "parser errors: {:?}", p.errors);
        jules_borrowck(&prog)
            .items
            .into_iter()
            .filter(|d| matches!(d.severity, Severity::Error))
            .count()
    }

    #[test]
    fn catches_write_while_borrowed() {
        let src = r#"
            fn main() {
                let mut x = 1;
                let r = &x;
                x = 2;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn catches_conflicting_mut_borrow() {
        let src = r#"
            fn main() {
                let mut x = 1;
                let a = &mut x;
                let b = &x;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn catches_use_after_move() {
        let src = r#"
            fn main() {
                let x = 1;
                let y = x;
                let z = x;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn catches_move_while_borrowed() {
        let src = r#"
            fn main() {
                let x = 1;
                let r = &x;
                let y = x;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn catches_borrow_after_move() {
        let src = r#"
            fn main() {
                let x = 1;
                let y = x;
                let r = &x;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }

    #[test]
    fn catches_assign_through_immutable_ref() {
        let src = r#"
            fn main() {
                let mut x = 1;
                let r = &x;
                *r = 3;
            }
        "#;
        assert!(borrow_errors(src) >= 1);
    }
}
