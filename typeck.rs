// =============================================================================
// jules/src/typeck.rs
//
// Type System / Type Checker / Type Inference for the Jules programming language.
//
// Responsibilities
// ────────────────
//   1. Resolve `Type` nodes — convert `Type::Named("Foo")` to concrete types by
//      looking up struct, component, enum, and agent declarations.
//
//   2. Validate operations — check that tensor algebra (`@`, `.*`, `++`),
//      vector/matrix math (`vec3 * mat3`), and scalar arithmetic are applied
//      only to operands of compatible types.  Infer the result type.
//
//   3. Infer `Type::Infer` / `_` — run a simple Hindley-Milner–style unification
//      solver over inference variables so that `let x: _ = A @ B` resolves `x`
//      to `tensor<f32>[M, K]`.
//
//   4. Check function signatures — verify that call sites provide the right
//      number and types of arguments, that generic bounds are satisfied, and
//      that return types are respected.
//
//   5. Component access correctness — verify that field names accessed on the
//      entity loop variable (`entity.position`) match declared component fields
//      and that the component appears in the system / entity-for query.
//
//   6. System semantics — check that `system` parameters have explicit types,
//      that write-accessed components are in `with` (or can be inferred), and
//      flag potential data races when two systems write the same component.
//
//   7. Neural-network model consistency — verify that layer shapes are compatible
//      (e.g., Dense output matches the next Dense input when both are explicit),
//      dropout rates are in [0, 1), and attention head dimensions are positive.
//
//   8. Span-aware diagnostics — every error and warning carries the originating
//      `Span` so the compiler front-end can point at the exact source location.
//
// Architecture
// ────────────
//   TypeCk            — the top-level checker; holds the global symbol table
//                        and a `Diagnostics` accumulator.
//   TyEnv             — per-scope type environment (variable → inferred type).
//   InferCtx          — inference-variable context (substitution map for `_`).
//   Diagnostics       — accumulates `Diagnostic` values (errors + warnings).
//   Ty                — the checker's internal type representation (richer than
//                        the parser's `ast::Type`).
// =============================================================================

use std::collections::HashMap;

use crate::ast::{
    AccessMode, AgentDecl, BinOpKind, Block, ComponentAccess, ComponentDecl, DimExpr, ElemType,
    EnumDecl, Expr, FnDecl, GenericParam, Item, ModelDecl, ModelLayer, Param, Pattern, Program,
    StructDecl, StructField, SystemDecl, TrainDecl, Type, UnOpKind, VecFamily, VecSize,
};
use crate::lexer::Span;

// =============================================================================
// INTERNAL TYPE REPRESENTATION
// =============================================================================

/// The type-checker's internal, fully-resolved type.
///
/// This parallels `ast::Type` but:
///   • Named types are resolved to their declaration.
///   • Inference variables carry a unique ID instead of just being `Infer`.
///   • Shape dimensions are evaluated to concrete `u64` where possible.
#[derive(Debug, Clone, PartialEq)]
pub enum Ty {
    // ── Scalars ──────────────────────────────────────────────────────────────
    Scalar(ElemType),
    Bool,
    Str,
    Unit,  // `()`
    Never, // `!`

    // ── Tensors (Feature 1) ───────────────────────────────────────────────────
    /// `tensor<elem>[d0, d1, …]`
    Tensor {
        elem: ElemType,
        shape: Vec<Dim>,
    },

    // ── SIMD vector / matrix types (Feature 2) ────────────────────────────────
    Vec {
        size: VecSize,
        family: VecFamily,
    },
    Mat {
        size: VecSize,
    }, // always float
    Quat,

    // ── Compound ─────────────────────────────────────────────────────────────
    Tuple(Vec<Ty>),
    Array {
        elem: Box<Ty>,
        len: u64,
    },
    Slice(Box<Ty>),
    Ref {
        mutable: bool,
        inner: Box<Ty>,
    },
    Option(Box<Ty>),
    FnPtr {
        params: Vec<Ty>,
        ret: Box<Ty>,
    },

    // ── User-defined ──────────────────────────────────────────────────────────
    /// A resolved struct or component (name preserved for diagnostics).
    Struct(String),
    /// A resolved enum variant's parent type.
    Enum(String),
    /// An ECS component type (subset of `Struct`).
    Component(String),
    /// The ECS world handle: `world`.
    World,

    // ── Inference variable ────────────────────────────────────────────────────
    /// An unresolved `_` slot; ID is unique within the current `InferCtx`.
    Infer(u32),
}

/// A single dimension in a tensor shape — may still be symbolic at check time.
#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    /// Concrete size: `[128, 128]`
    Lit(u64),
    /// Named compile-time constant: `[N]`
    Named(String),
    /// Runtime-dynamic: `[_]`
    Dynamic,
}

impl Ty {
    /// True if this type is numeric (eligible for arithmetic ops).
    pub fn is_numeric(&self) -> bool {
        matches!(self,
            Ty::Scalar(e) if !matches!(e, ElemType::Bool)
        )
    }

    /// True if this type is a floating-point scalar.
    pub fn is_float_scalar(&self) -> bool {
        matches!(self, Ty::Scalar(e) if e.is_float())
    }

    /// True for any tensor or SIMD vector / matrix.
    pub fn is_tensor_like(&self) -> bool {
        matches!(
            self,
            Ty::Tensor { .. } | Ty::Vec { .. } | Ty::Mat { .. } | Ty::Quat
        )
    }

    /// True if this type can be element-wise multiplied (Hadamard).
    pub fn supports_hadamard(&self) -> bool {
        matches!(self, Ty::Tensor { .. } | Ty::Vec { .. } | Ty::Mat { .. })
    }

    /// True for types that can accumulate a gradient (`@grad` marker).
    pub fn supports_grad(&self) -> bool {
        match self {
            Ty::Scalar(e) => e.is_float(),
            Ty::Tensor { elem, .. } => elem.is_float(),
            Ty::Vec { family, .. } => *family == VecFamily::Float,
            Ty::Mat { .. } | Ty::Quat => true,
            _ => false,
        }
    }

    /// A human-readable name for error messages.
    pub fn display(&self) -> String {
        match self {
            Ty::Scalar(e) => format!("{:?}", e).to_lowercase(),
            Ty::Bool => "bool".into(),
            Ty::Str => "str".into(),
            Ty::Unit => "()".into(),
            Ty::Never => "!".into(),
            Ty::Tensor { elem, shape } => {
                let dims: Vec<_> = shape
                    .iter()
                    .map(|d| match d {
                        Dim::Lit(n) => n.to_string(),
                        Dim::Named(s) => s.clone(),
                        Dim::Dynamic => "_".into(),
                    })
                    .collect();
                format!("tensor<{:?}>[{}]", elem, dims.join(", ")).to_lowercase()
            }
            Ty::Vec { size, family } => format!("{:?}{}", family, size.lanes()).to_lowercase(),
            Ty::Mat { size } => format!("mat{}", size.lanes()),
            Ty::Quat => "quat".into(),
            Ty::Tuple(ts) => {
                let inner: Vec<_> = ts.iter().map(|t| t.display()).collect();
                format!("({})", inner.join(", "))
            }
            Ty::Array { elem, len } => format!("[{}; {}]", elem.display(), len),
            Ty::Slice(inner) => format!("[{}]", inner.display()),
            Ty::Ref { mutable, inner } => {
                if *mutable {
                    format!("&mut {}", inner.display())
                } else {
                    format!("&{}", inner.display())
                }
            }
            Ty::Option(inner) => format!("Option<{}>", inner.display()),
            Ty::FnPtr { params, ret } => {
                let ps: Vec<_> = params.iter().map(|p| p.display()).collect();
                format!("fn({}) -> {}", ps.join(", "), ret.display())
            }
            Ty::Struct(name) => name.clone(),
            Ty::Enum(name) => name.clone(),
            Ty::Component(name) => name.clone(),
            Ty::World => "world".into(),
            Ty::Infer(id) => format!("_#{}", id),
        }
    }
}

// =============================================================================
// DIAGNOSTICS
// =============================================================================

/// Severity of a diagnostic message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

/// A single compiler diagnostic (error or warning) with source location.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub span: Span,
    pub message: String,
    /// Optional secondary "note" labels attached to other spans.
    pub notes: Vec<(Span, String)>,
}

impl Diagnostic {
    pub fn error(span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Error,
            span,
            message: message.into(),
            notes: vec![],
        }
    }
    pub fn warning(span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Warning,
            span,
            message: message.into(),
            notes: vec![],
        }
    }
    pub fn note(span: Span, message: impl Into<String>) -> Self {
        Diagnostic {
            severity: Severity::Note,
            span,
            message: message.into(),
            notes: vec![],
        }
    }
    pub fn with_note(mut self, span: Span, msg: impl Into<String>) -> Self {
        self.notes.push((span, msg.into()));
        self
    }

    /// True if this diagnostic would prevent code generation.
    pub fn is_fatal(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// Accumulates diagnostics from all checker passes.
#[derive(Debug, Default)]
pub struct Diagnostics {
    pub items: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn push(&mut self, d: Diagnostic) {
        self.items.push(d);
    }
    pub fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.push(Diagnostic::error(span, msg));
    }
    pub fn warning(&mut self, span: Span, msg: impl Into<String>) {
        self.push(Diagnostic::warning(span, msg));
    }
    pub fn note(&mut self, span: Span, msg: impl Into<String>) {
        self.push(Diagnostic::note(span, msg));
    }
    pub fn has_errors(&self) -> bool {
        self.items.iter().any(|d| d.is_fatal())
    }
    pub fn error_count(&self) -> usize {
        self.items.iter().filter(|d| d.is_fatal()).count()
    }
}

// =============================================================================
// TYPE INFERENCE CONTEXT
// =============================================================================

/// A simple substitution-based inference context.
///
/// Inference variables are numbered from 0 upwards.  `unify` adds a binding
/// `var → ty` to the map; `resolve` follows the chain to its root.
#[derive(Debug, Default)]
pub struct InferCtx {
    next_var: u32,
    /// Substitution map: inference-var ID → resolved type.
    subst: HashMap<u32, Ty>,
}

impl InferCtx {
    /// Allocate a fresh inference variable.
    pub fn fresh(&mut self) -> Ty {
        let id = self.next_var;
        self.next_var += 1;
        Ty::Infer(id)
    }

    /// Walk the substitution chain until we reach a non-variable type (or an
    /// unresolved variable).
    pub fn resolve(&self, ty: &Ty) -> Ty {
        match ty {
            Ty::Infer(id) => match self.subst.get(id) {
                Some(t) => self.resolve(t),
                None => ty.clone(),
            },
            _ => ty.clone(),
        }
    }

    /// Attempt to unify `a` and `b`, recording the binding.
    /// Returns `true` on success, `false` on mismatch.
    pub fn unify(&mut self, a: &Ty, b: &Ty) -> bool {
        let a = self.resolve(a);
        let b = self.resolve(b);
        match (&a, &b) {
            (Ty::Infer(id), other) | (other, Ty::Infer(id)) => {
                // Occurs check: don't bind a var to itself.
                if matches!(other, Ty::Infer(oid) if oid == id) {
                    return true;
                }
                self.subst.insert(*id, other.clone());
                true
            }
            // Tensors unify if elem matches and ranks match.
            (
                Ty::Tensor {
                    elem: ea,
                    shape: sa,
                },
                Ty::Tensor {
                    elem: eb,
                    shape: sb,
                },
            ) => {
                ea == eb
                    && sa.len() == sb.len()
                    && sa.iter().zip(sb).all(|(a, b)| match (a, b) {
                        (Dim::Lit(x), Dim::Lit(y)) => x == y,
                        (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
                        (Dim::Named(x), Dim::Named(y)) => x == y,
                        _ => false,
                    })
            }
            // Tuples unify element-wise.
            (Ty::Tuple(ts_a), Ty::Tuple(ts_b)) => {
                ts_a.len() == ts_b.len() && ts_a.iter().zip(ts_b).all(|(a, b)| self.unify(a, b))
            }
            // Everything else must be structurally equal.
            _ => a == b,
        }
    }
}

// =============================================================================
// TYPE ENVIRONMENT (per-scope variable bindings)
// =============================================================================

/// A scoped variable → type binding table.
/// Scopes are nested with `push_scope` / `pop_scope`.
#[derive(Debug, Default)]
pub struct TyEnv {
    /// Stack of scopes; the last entry is the innermost.
    scopes: Vec<HashMap<String, Ty>>,
}

impl TyEnv {
    pub fn new() -> Self {
        TyEnv {
            scopes: vec![HashMap::new()],
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Bind a name in the innermost scope.
    pub fn bind(&mut self, name: impl Into<String>, ty: Ty) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), ty);
        }
    }

    /// Look up a name, searching from inner to outer scope.
    pub fn lookup(&self, name: &str) -> Option<&Ty> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }
}

// =============================================================================
// GLOBAL SYMBOL TABLE
// =============================================================================

/// Resolved information about a struct / component declaration.
#[derive(Debug, Clone)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<(String, Ty)>,
    /// True when this is a `component` rather than a plain `struct`.
    pub is_component: bool,
}

impl StructInfo {
    /// Look up a field by name.
    pub fn field_ty(&self, name: &str) -> Option<&Ty> {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, ty)| ty)
    }
}

/// Resolved information about a function declaration.
#[derive(Debug, Clone)]
pub struct FnInfo {
    pub name: String,
    pub params: Vec<(String, Ty)>,
    pub ret: Ty,
    pub generics: Vec<GenericParam>,
}

/// The program-level symbol table populated during the first pass over items.
#[derive(Debug, Default)]
pub struct SymbolTable {
    pub structs: HashMap<String, StructInfo>,
    pub fns: HashMap<String, FnInfo>,
    pub agents: HashMap<String, AgentDecl>,
    pub models: HashMap<String, ModelDecl>,
}

impl SymbolTable {
    pub fn lookup_struct(&self, name: &str) -> Option<&StructInfo> {
        self.structs.get(name)
    }
    pub fn lookup_fn(&self, name: &str) -> Option<&FnInfo> {
        self.fns.get(name)
    }
}

// =============================================================================
// MAIN TYPE CHECKER
// =============================================================================

/// The main Jules type-checker.
///
/// Usage:
/// ```rust,ignore
/// let mut ck = TypeCk::new();
/// ck.check_program(&ast);
/// if ck.diag.has_errors() {
///     for d in &ck.diag.items { eprintln!("{}: {}", d.span, d.message); }
/// }
/// ```
pub struct TypeCk {
    pub diag: Diagnostics,
    pub symbols: SymbolTable,
    pub infer: InferCtx,
}

impl TypeCk {
    pub fn new() -> Self {
        TypeCk {
            diag: Diagnostics::default(),
            symbols: SymbolTable::default(),
            infer: InferCtx::default(),
        }
    }

    // =========================================================================
    // PASS 1 — Build the global symbol table
    // =========================================================================

    /// Walk all top-level items and register declarations in the symbol table.
    /// This is a "shallow" pass — bodies are not checked yet.
    pub fn collect_items(&mut self, program: &Program) {
        for item in &program.items {
            self.collect_item(item);
        }
    }

    fn collect_item(&mut self, item: &Item) {
        match item {
            Item::Struct(s) => self.collect_struct(s, false),
            Item::Component(c) => self.collect_component(c),
            Item::Fn(f) => self.collect_fn(f),
            Item::Agent(a) => {
                self.symbols.agents.insert(a.name.clone(), a.clone());
            }
            Item::Model(m) => {
                self.symbols.models.insert(m.name.clone(), m.clone());
            }
            Item::Mod {
                items: Some(inner), ..
            } => {
                for i in inner {
                    self.collect_item(i);
                }
            }
            _ => {}
        }
    }

    fn collect_struct(&mut self, s: &StructDecl, is_component: bool) {
        let fields = s
            .fields
            .iter()
            .map(|f| (f.name.clone(), self.lower_ast_type(&f.ty, f.span)))
            .collect();
        self.symbols.structs.insert(
            s.name.clone(),
            StructInfo {
                name: s.name.clone(),
                fields,
                is_component,
            },
        );
    }

    fn collect_component(&mut self, c: &ComponentDecl) {
        let fields = c
            .fields
            .iter()
            .map(|f| (f.name.clone(), self.lower_ast_type(&f.ty, f.span)))
            .collect();
        self.symbols.structs.insert(
            c.name.clone(),
            StructInfo {
                name: c.name.clone(),
                fields,
                is_component: true,
            },
        );
    }

    fn collect_fn(&mut self, f: &FnDecl) {
        let params = f
            .params
            .iter()
            .map(|p| {
                let ty =
                    p.ty.as_ref()
                        .map(|t| self.lower_ast_type(t, p.span))
                        .unwrap_or_else(|| self.infer.fresh());
                (p.name.clone(), ty)
            })
            .collect();
        let ret = f
            .ret_ty
            .as_ref()
            .map(|t| self.lower_ast_type(t, f.span))
            .unwrap_or(Ty::Unit);
        self.symbols.fns.insert(
            f.name.clone(),
            FnInfo {
                name: f.name.clone(),
                params,
                ret,
                generics: f.generics.clone(),
            },
        );
    }

    // =========================================================================
    // AST TYPE → INTERNAL TYPE
    // =========================================================================

    /// Lower a parsed `ast::Type` into the checker's internal `Ty`.
    pub fn lower_ast_type(&mut self, ty: &Type, span: Span) -> Ty {
        match ty {
            Type::Scalar(e) => Ty::Scalar(e.clone()),
            Type::Infer => self.infer.fresh(),
            Type::Never => Ty::Never,

            Type::Tensor { elem, shape } => {
                let dims = shape.iter().map(|d| self.lower_dim(d)).collect();
                Ty::Tensor {
                    elem: elem.clone(),
                    shape: dims,
                }
            }

            Type::Vec { size, family } => Ty::Vec {
                size: *size,
                family: *family,
            },
            Type::Mat { size } => Ty::Mat { size: *size },
            Type::Quat => Ty::Quat,

            Type::Named(name) => {
                match name.as_str() {
                    "world" | "World" => Ty::World,
                    "bool" => Ty::Bool,
                    "str" | "String" => Ty::Str,
                    _ => {
                        if self.symbols.structs.contains_key(name) {
                            let info = &self.symbols.structs[name];
                            if info.is_component {
                                Ty::Component(name.clone())
                            } else {
                                Ty::Struct(name.clone())
                            }
                        } else {
                            // Not yet declared — emit an error, return infer.
                            self.diag.error(span, format!("unknown type `{}`", name));
                            self.infer.fresh()
                        }
                    }
                }
            }

            Type::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.lower_ast_type(t, span)).collect()),

            Type::Array { elem, len } => {
                // Try to evaluate the length expression.
                if let Some(n) = eval_const_expr(len) {
                    Ty::Array {
                        elem: Box::new(self.lower_ast_type(elem, span)),
                        len: n,
                    }
                } else {
                    self.diag
                        .error(span, "array length must be a compile-time constant");
                    self.infer.fresh()
                }
            }

            Type::Slice(inner) => Ty::Slice(Box::new(self.lower_ast_type(inner, span))),

            Type::Ref { mutable, inner } => Ty::Ref {
                mutable: *mutable,
                inner: Box::new(self.lower_ast_type(inner, span)),
            },

            Type::Option(inner) => Ty::Option(Box::new(self.lower_ast_type(inner, span))),

            Type::FnPtr { params, ret } => {
                let ps = params
                    .iter()
                    .map(|t| self.lower_ast_type(t, span))
                    .collect();
                let r = self.lower_ast_type(ret, span);
                Ty::FnPtr {
                    params: ps,
                    ret: Box::new(r),
                }
            }
            Type::Result { ok: _, err: _ } => self.infer.fresh(),
        }
    }

    fn lower_dim(&self, d: &DimExpr) -> Dim {
        match d {
            DimExpr::Lit(n) => Dim::Lit(*n),
            DimExpr::Named(s) => Dim::Named(s.clone()),
            DimExpr::Dynamic => Dim::Dynamic,
            DimExpr::Expr(e) => eval_const_expr(e).map(Dim::Lit).unwrap_or(Dim::Dynamic),
        }
    }

    // =========================================================================
    // PASS 2 — Check all items
    // =========================================================================

    pub fn check_program(&mut self, program: &Program) {
        // First pass: collect all declarations so forward references work.
        self.collect_items(program);

        // Second pass: type-check bodies.
        for item in &program.items {
            self.check_item(item);
        }

        // Post-pass: cross-system data-race analysis.
        let systems: Vec<_> = program
            .items
            .iter()
            .filter_map(|i| {
                if let Item::System(s) = i {
                    Some(s)
                } else {
                    None
                }
            })
            .collect();
        self.check_system_data_races(&systems);
    }

    fn check_item(&mut self, item: &Item) {
        match item {
            Item::Fn(f) => self.check_fn(f),
            Item::System(s) => self.check_system(s),
            Item::Struct(_) => { /* fields validated in collect */ }
            Item::Component(_) => { /* fields validated in collect */ }
            Item::Model(m) => self.check_model(m),
            Item::Train(t) => self.check_train(t),
            Item::Agent(a) => self.check_agent(a),
            Item::Const(c) => {
                let mut env = TyEnv::new();
                let expected = self.lower_ast_type(&c.ty, c.span);
                let actual = self.check_expr(&c.value, &mut env);
                self.expect_ty(&expected, &actual, c.span, "const initialiser");
            }
            Item::Mod {
                items: Some(inner), ..
            } => {
                for i in inner {
                    self.check_item(i);
                }
            }
            _ => {}
        }
    }

    // =========================================================================
    // FUNCTION CHECKING
    // =========================================================================

    fn check_fn(&mut self, f: &FnDecl) {
        let mut env = TyEnv::new();
        env.push_scope();

        // Bind parameters.
        for param in &f.params {
            let ty = match &param.ty {
                Some(t) => self.lower_ast_type(t, param.span),
                None => {
                    // Unannotated params are allowed for closures; warn for fns.
                    self.diag.warning(
                        param.span,
                        format!(
                            "parameter `{}` has no type annotation; inferred as `_`",
                            param.name
                        ),
                    );
                    self.infer.fresh()
                }
            };
            env.bind(param.name.clone(), ty);
        }

        // Check body.
        if let Some(body) = &f.body {
            let body_ty = self.check_block(body, &mut env);
            let ret_ty = f
                .ret_ty
                .as_ref()
                .map(|t| self.lower_ast_type(t, f.span))
                .unwrap_or(Ty::Unit);

            if !self.infer.unify(&ret_ty, &body_ty) {
                self.diag.error(
                    f.span,
                    format!(
                        "function `{}` declared to return `{}` but body returns `{}`",
                        f.name,
                        self.infer.resolve(&ret_ty).display(),
                        self.infer.resolve(&body_ty).display(),
                    ),
                );
            }
        }

        // Generic bound satisfaction (simplified: warn if bounds list is non-empty
        // and we don't have a full trait system yet).
        for gp in &f.generics {
            if !gp.bounds.is_empty() {
                self.diag.note(
                    gp.span,
                    format!(
                        "generic `{}` has bounds {:?}; full trait checking not yet implemented",
                        gp.name, gp.bounds
                    ),
                );
            }
        }

        env.pop_scope();
    }

    // =========================================================================
    // SYSTEM CHECKING (Feature 2 — Game Simulation)
    // =========================================================================

    fn check_system(&mut self, s: &SystemDecl) {
        let mut env = TyEnv::new();
        env.push_scope();

        // All system parameters MUST have explicit types (they become uniforms).
        for param in &s.params {
            if param.ty.is_none() {
                self.diag.error(
                    param.span,
                    format!(
                        "system parameter `{}` must have an explicit type annotation \
                     (system parameters become GPU uniforms / push constants)",
                        param.name
                    ),
                );
            }
            let ty = param
                .ty
                .as_ref()
                .map(|t| self.lower_ast_type(t, param.span))
                .unwrap_or_else(|| self.infer.fresh());
            env.bind(param.name.clone(), ty);
        }

        // Bind `world` as the ECS world handle.
        env.bind("world", Ty::World);

        // Check body and collect component accesses.
        let _body_ty = self.check_block(&s.body, &mut env);

        // Validate any explicit query components are declared.
        if let Some(q) = &s.explicit_query {
            for comp in q.with.iter().chain(q.without.iter()) {
                if !self.symbols.structs.contains_key(comp.as_str()) {
                    self.diag.error(
                        q.span,
                        format!("query references unknown component `{}`", comp),
                    );
                }
            }
        }

        env.pop_scope();
    }

    /// Cross-system data-race analysis: warn when two systems both access the
    /// same component and at least one of them writes it.
    fn check_system_data_races(&mut self, systems: &[&SystemDecl]) {
        for i in 0..systems.len() {
            for j in (i + 1)..systems.len() {
                let a = systems[i];
                let b = systems[j];
                for acc_a in &a.accesses {
                    for acc_b in &b.accesses {
                        if acc_a.component == acc_b.component
                            && (acc_a.mode.is_write() || acc_b.mode.is_write())
                        {
                            self.diag.push(
                                Diagnostic::warning(
                                    a.span,
                                    format!(
                                        "potential data race on component `{}`: \
                                     system `{}` ({:?}) and system `{}` ({:?}) \
                                     both access it and at least one writes",
                                        acc_a.component, a.name, acc_a.mode, b.name, acc_b.mode,
                                    ),
                                )
                                .with_note(b.span, "second system here"),
                            );
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // EXPRESSION TYPE CHECKING
    // =========================================================================

    /// Check an expression and return its inferred type.
    pub fn check_expr(&mut self, expr: &Expr, env: &mut TyEnv) -> Ty {
        match expr {
            // ── Literals ─────────────────────────────────────────────────────
            Expr::IntLit { .. } => Ty::Scalar(ElemType::I32), // default int
            Expr::FloatLit { .. } => Ty::Scalar(ElemType::F32), // default float
            Expr::BoolLit { .. } => Ty::Bool,
            Expr::StrLit { .. } => Ty::Str,

            // ── Variable lookup ───────────────────────────────────────────────
            Expr::Ident { span, name } => match env.lookup(name) {
                Some(ty) => ty.clone(),
                None => {
                    self.diag
                        .error(*span, format!("use of undeclared variable `{}`", name));
                    self.infer.fresh()
                }
            },

            Expr::Path { span, segments } => {
                let name = segments.join("::");
                match env.lookup(&name) {
                    Some(ty) => ty.clone(),
                    None => {
                        // Could be a function reference.
                        if let Some(fi) = self
                            .symbols
                            .fns
                            .get(segments.last().map(|s| s.as_str()).unwrap_or(""))
                        {
                            Ty::FnPtr {
                                params: fi.params.iter().map(|(_, t)| t.clone()).collect(),
                                ret: Box::new(fi.ret.clone()),
                            }
                        } else {
                            self.diag.error(*span, format!("unknown path `{}`", name));
                            self.infer.fresh()
                        }
                    }
                }
            }

            // ── Vector constructor ─────────────────────────────────────────────
            Expr::VecCtor { span, size, elems } => {
                let n = size.lanes() as usize;
                if elems.len() != n {
                    self.diag.error(
                        *span,
                        format!(
                            "vec{} constructor expects {} elements, got {}",
                            n,
                            n,
                            elems.len()
                        ),
                    );
                }
                for e in elems {
                    let ty = self.check_expr(e, env);
                    if !ty.is_float_scalar() {
                        self.diag.warning(
                            e.span(),
                            format!(
                                "vec constructor element should be f32; got `{}`",
                                ty.display()
                            ),
                        );
                    }
                }
                Ty::Vec {
                    size: *size,
                    family: VecFamily::Float,
                }
            }

            // ── Array literal ──────────────────────────────────────────────────
            Expr::ArrayLit { span, elems } => {
                if elems.is_empty() {
                    return Ty::Array {
                        elem: Box::new(self.infer.fresh()),
                        len: 0,
                    };
                }
                let first_ty = self.check_expr(&elems[0], env);
                for e in &elems[1..] {
                    let ty = self.check_expr(e, env);
                    if !self.infer.unify(&first_ty, &ty) {
                        self.diag.error(
                            *span,
                            format!(
                                "array literal has inconsistent element types: `{}` vs `{}`",
                                first_ty.display(),
                                ty.display()
                            ),
                        );
                    }
                }
                Ty::Array {
                    elem: Box::new(first_ty),
                    len: elems.len() as u64,
                }
            }

            // ── Binary operators ───────────────────────────────────────────────
            Expr::BinOp { span, op, lhs, rhs } => self.check_binop(*span, *op, lhs, rhs, env),

            // ── Unary operators ────────────────────────────────────────────────
            Expr::UnOp { span, op, expr } => {
                let ty = self.check_expr(expr, env);
                match op {
                    UnOpKind::Neg => {
                        if !ty.is_numeric() && !ty.is_tensor_like() {
                            self.diag.error(
                                *span,
                                format!(
                                    "unary `-` requires a numeric or tensor type, got `{}`",
                                    ty.display()
                                ),
                            );
                        }
                        ty
                    }
                    UnOpKind::Not => {
                        if !matches!(ty, Ty::Bool) && !ty.is_numeric() {
                            self.diag.error(
                                *span,
                                format!(
                                    "unary `!`/`~` requires `bool` or integer, got `{}`",
                                    ty.display()
                                ),
                            );
                        }
                        if matches!(ty, Ty::Bool) {
                            Ty::Bool
                        } else {
                            ty
                        }
                    }
                    UnOpKind::Deref => match ty {
                        Ty::Ref { inner, .. } => *inner,
                        _ => {
                            self.diag
                                .error(*span, format!("cannot dereference `{}`", ty.display()));
                            self.infer.fresh()
                        }
                    },
                    UnOpKind::Ref => Ty::Ref {
                        mutable: false,
                        inner: Box::new(ty),
                    },
                    UnOpKind::RefMut => Ty::Ref {
                        mutable: true,
                        inner: Box::new(ty),
                    },
                }
            }

            // ── Assignment ────────────────────────────────────────────────────
            Expr::Assign {
                span,
                op: _,
                target,
                value,
            } => {
                let t_ty = self.check_expr(target, env);
                let v_ty = self.check_expr(value, env);
                if !self.infer.unify(&t_ty, &v_ty) {
                    self.diag.error(
                        *span,
                        format!(
                            "type mismatch in assignment: cannot assign `{}` to `{}`",
                            v_ty.display(),
                            t_ty.display()
                        ),
                    );
                }
                Ty::Unit
            }

            // ── Field access ───────────────────────────────────────────────────
            Expr::Field {
                span,
                object,
                field,
            } => self.check_field_access(*span, object, field, env),

            // ── Index ──────────────────────────────────────────────────────────
            Expr::Index {
                span,
                object,
                indices,
            } => {
                let obj_ty = self.check_expr(object, env);
                for idx in indices {
                    let i_ty = self.check_expr(idx, env);
                    if !matches!(
                        i_ty,
                        Ty::Scalar(
                            ElemType::I32
                                | ElemType::I64
                                | ElemType::U32
                                | ElemType::U64
                                | ElemType::Usize
                        )
                    ) {
                        self.diag.warning(
                            *span,
                            format!("index should be an integer type, got `{}`", i_ty.display()),
                        );
                    }
                }
                match &obj_ty {
                    Ty::Tensor { elem, shape } => {
                        if indices.len() != shape.len() {
                            self.diag.error(
                                *span,
                                format!(
                                    "tensor has rank {} but {} indices provided",
                                    shape.len(),
                                    indices.len()
                                ),
                            );
                        }
                        Ty::Scalar(elem.clone())
                    }
                    Ty::Array { elem, .. } | Ty::Slice(elem) => *elem.clone(),
                    _ => {
                        self.diag
                            .error(*span, format!("cannot index into `{}`", obj_ty.display()));
                        self.infer.fresh()
                    }
                }
            }

            // ── Call ───────────────────────────────────────────────────────────
            Expr::Call {
                span,
                func,
                args,
                named,
            } => self.check_call(*span, func, args, named, env),

            // ── Method call ───────────────────────────────────────────────────
            Expr::MethodCall {
                span,
                receiver,
                method,
                args,
            } => self.check_method_call(*span, receiver, method, args, env),

            // ── Tensor algebra (Feature 1) ─────────────────────────────────────
            Expr::MatMul { span, lhs, rhs } => self.check_matmul(*span, lhs, rhs, env),

            Expr::HadamardMul { span, lhs, rhs } | Expr::HadamardDiv { span, lhs, rhs } => {
                self.check_hadamard(*span, lhs, rhs, env)
            }

            Expr::TensorConcat { span, lhs, rhs } => self.check_tensor_concat(*span, lhs, rhs, env),

            // ── Kronecker product A @@ B → [m*p, n*q] where A:[m,n] B:[p,q] ──
            Expr::KronProd { span, lhs, rhs } => {
                let l = self.check_expr(lhs, env);
                let r = self.check_expr(rhs, env);
                if let (
                    Ty::Tensor {
                        elem: ea,
                        shape: sa,
                    },
                    Ty::Tensor {
                        elem: eb,
                        shape: sb,
                    },
                ) = (&l, &r)
                {
                    if ea != eb {
                        self.diag
                            .error(*span, "Kronecker product: element type mismatch");
                    }
                    if sa.len() != 2 || sb.len() != 2 {
                        self.diag
                            .error(*span, "Kronecker product (`@@`) requires 2-D tensors");
                        return self.infer.fresh();
                    }
                    let d0 = match (&sa[0], &sb[0]) {
                        (Dim::Lit(a), Dim::Lit(b)) => Dim::Lit(a * b),
                        _ => Dim::Dynamic,
                    };
                    let d1 = match (&sa[1], &sb[1]) {
                        (Dim::Lit(a), Dim::Lit(b)) => Dim::Lit(a * b),
                        _ => Dim::Dynamic,
                    };
                    return Ty::Tensor {
                        elem: ea.clone(),
                        shape: vec![d0, d1],
                    };
                }
                self.diag.error(
                    *span,
                    format!(
                        "`@@` (Kronecker product) requires 2-D tensor operands; got `{}` and `{}`",
                        l.display(),
                        r.display()
                    ),
                );
                self.infer.fresh()
            }

            // ── Outer product a ^* b → [m, n] where a:[m] b:[n] ──────────────
            Expr::OuterProd { span, lhs, rhs } => {
                let l = self.check_expr(lhs, env);
                let r = self.check_expr(rhs, env);
                let (elem, d0, d1) = match (&l, &r) {
                    (
                        Ty::Tensor {
                            elem: ea,
                            shape: sa,
                        },
                        Ty::Tensor {
                            elem: eb,
                            shape: sb,
                        },
                    ) if ea == eb && sa.len() == 1 && sb.len() == 1 => {
                        (ea.clone(), sa[0].clone(), sb[0].clone())
                    }
                    (
                        Ty::Vec {
                            family: fa,
                            size: sa,
                        },
                        Ty::Vec {
                            family: fb,
                            size: sb,
                        },
                    ) if fa == fb => {
                        let elem = if *fa == VecFamily::Float {
                            ElemType::F32
                        } else {
                            ElemType::I32
                        };
                        let m = sa.lanes() as u64;
                        let n = sb.lanes() as u64;
                        return Ty::Tensor {
                            elem,
                            shape: vec![Dim::Lit(m), Dim::Lit(n)],
                        };
                    }
                    _ => {
                        self.diag.error(*span, format!(
                            "`^*` (outer product) requires 1-D tensor or vector operands; got `{}` and `{}`",
                            l.display(), r.display()
                        ));
                        return self.infer.fresh();
                    }
                };
                Ty::Tensor {
                    elem,
                    shape: vec![d0, d1],
                }
            }

            Expr::Grad { span, inner } => {
                let ty = self.check_expr(inner, env);
                if !ty.supports_grad() {
                    self.diag.error(
                        *span,
                        format!(
                            "`grad` requires a float tensor or vector type, got `{}`",
                            ty.display()
                        ),
                    );
                }
                ty
            }

            Expr::Pow { span, base, exp } => {
                let b_ty = self.check_expr(base, env);
                let e_ty = self.check_expr(exp, env);
                if !b_ty.is_numeric() && !b_ty.is_tensor_like() {
                    self.diag.error(
                        *span,
                        format!(
                            "`**` base must be numeric or tensor, got `{}`",
                            b_ty.display()
                        ),
                    );
                }
                if !e_ty.is_numeric() {
                    self.diag.error(
                        *span,
                        format!("`**` exponent must be numeric, got `{}`", e_ty.display()),
                    );
                }
                b_ty
            }

            // ── Range ──────────────────────────────────────────────────────────
            Expr::Range { span, lo, hi, .. } => {
                let mut bound_ty: Option<Ty> = None;
                for bound in lo.iter().chain(hi.iter()) {
                    let ty = self.check_expr(bound, env);
                    if let Some(ref prev) = bound_ty {
                        if !self.infer.unify(prev, &ty) {
                            self.diag
                                .error(*span, "range bounds have incompatible types");
                        }
                    } else {
                        bound_ty = Some(ty);
                    }
                }
                // Range<T> — return as a slice-like type for now.
                Ty::Slice(Box::new(bound_ty.unwrap_or_else(|| self.infer.fresh())))
            }

            // ── Cast ───────────────────────────────────────────────────────────
            Expr::Cast { span, expr, ty } => {
                let src = self.check_expr(expr, env);
                let dst = self.lower_ast_type(ty, *span);
                // Allow numeric widening / narrowing; warn on lossy casts.
                if let (Ty::Scalar(se), Ty::Scalar(de)) = (&src, &dst) {
                    if se.byte_size() > de.byte_size() {
                        self.diag.warning(
                            *span,
                            format!(
                                "narrowing cast from `{}` to `{}`; may lose precision",
                                src.display(),
                                dst.display()
                            ),
                        );
                    }
                }
                dst
            }

            // ── If expression ──────────────────────────────────────────────────
            Expr::IfExpr {
                span,
                cond,
                then,
                else_,
            } => {
                let cond_ty = self.check_expr(cond, env);
                if !matches!(cond_ty, Ty::Bool) {
                    self.diag.error(
                        *span,
                        format!("`if` condition must be `bool`, got `{}`", cond_ty.display()),
                    );
                }
                let then_ty = self.check_block(then, env);
                if let Some(else_blk) = else_ {
                    let else_ty = self.check_block(else_blk, env);
                    if !self.infer.unify(&then_ty, &else_ty) {
                        self.diag.error(
                            *span,
                            format!(
                                "`if` branches have incompatible types: `{}` vs `{}`",
                                then_ty.display(),
                                else_ty.display()
                            ),
                        );
                    }
                }
                then_ty
            }

            // ── Closure ────────────────────────────────────────────────────────
            Expr::Closure {
                span: _,
                params,
                ret_ty,
                body,
            } => {
                env.push_scope();
                let param_tys: Vec<Ty> = params
                    .iter()
                    .map(|p| {
                        let ty =
                            p.ty.as_ref()
                                .map(|t| self.lower_ast_type(t, p.span))
                                .unwrap_or_else(|| self.infer.fresh());
                        env.bind(p.name.clone(), ty.clone());
                        ty
                    })
                    .collect();
                let body_ty = self.check_expr(body, env);
                env.pop_scope();
                let ret = ret_ty
                    .as_ref()
                    .map(|t| self.lower_ast_type(t, body.span()))
                    .unwrap_or(body_ty.clone());
                Ty::FnPtr {
                    params: param_tys,
                    ret: Box::new(ret),
                }
            }

            // ── Block expression ───────────────────────────────────────────────
            Expr::Block(b) => self.check_block(b, env),

            // ── Tuple ──────────────────────────────────────────────────────────
            Expr::Tuple { elems, .. } => {
                let tys = elems.iter().map(|e| self.check_expr(e, env)).collect();
                Ty::Tuple(tys)
            }

            // ── Struct literal ─────────────────────────────────────────────────
            Expr::StructLit { span, name, fields } => {
                if let Some(info) = self.symbols.structs.get(name).cloned() {
                    let is_component = info.is_component;
                    // Check each field.
                    for (fname, fexpr) in fields {
                        let expr_ty = self.check_expr(fexpr, env);
                        if let Some(expected) = info.field_ty(fname) {
                            let expected = expected.clone();
                            if !self.infer.unify(&expected, &expr_ty) {
                                self.diag.error(
                                    fexpr.span(),
                                    format!(
                                        "field `{}` of `{}` expects `{}`, got `{}`",
                                        fname,
                                        name,
                                        expected.display(),
                                        expr_ty.display()
                                    ),
                                );
                            }
                        } else {
                            self.diag
                                .error(*span, format!("`{}` has no field `{}`", name, fname));
                        }
                    }
                    if is_component {
                        Ty::Component(name.clone())
                    } else {
                        Ty::Struct(name.clone())
                    }
                } else {
                    self.diag.error(*span, format!("unknown struct `{}`", name));
                    self.infer.fresh()
                }
            }
        }
    }

    // =========================================================================
    // BINARY OPERATOR CHECKING
    // =========================================================================

    fn check_binop(
        &mut self,
        span: Span,
        op: BinOpKind,
        lhs: &Expr,
        rhs: &Expr,
        env: &mut TyEnv,
    ) -> Ty {
        let l = self.check_expr(lhs, env);
        let r = self.check_expr(rhs, env);

        match op {
            // Comparison → bool
            BinOpKind::Eq
            | BinOpKind::Ne
            | BinOpKind::Lt
            | BinOpKind::Le
            | BinOpKind::Gt
            | BinOpKind::Ge => {
                if !self.infer.unify(&l, &r) {
                    self.diag.error(
                        span,
                        format!(
                            "comparison of incompatible types `{}` and `{}`",
                            l.display(),
                            r.display()
                        ),
                    );
                }
                Ty::Bool
            }

            // Logical → bool
            BinOpKind::And | BinOpKind::Or => {
                for ty in [&l, &r] {
                    if !matches!(ty, Ty::Bool) {
                        self.diag.error(
                            span,
                            format!(
                                "logical operator requires `bool` operands, got `{}`",
                                ty.display()
                            ),
                        );
                    }
                }
                Ty::Bool
            }

            // Arithmetic
            BinOpKind::Add
            | BinOpKind::Sub
            | BinOpKind::Mul
            | BinOpKind::Div
            | BinOpKind::Rem
            | BinOpKind::FloorDiv => {
                // Vector × scalar or scalar × vector
                if let (Ty::Vec { size, family }, ref rty) | (ref rty, Ty::Vec { size, family }) =
                    (&l, &r)
                {
                    let size = *size;
                    let family = *family;
                    if rty.is_numeric() {
                        return Ty::Vec { size, family };
                    }
                }
                // vec3 × mat3 → vec3
                if let Ty::Vec {
                    size: vs,
                    family: VecFamily::Float,
                } = &l
                {
                    if let Ty::Mat { size: ms } = &r {
                        if vs == ms {
                            return Ty::Vec {
                                size: *vs,
                                family: VecFamily::Float,
                            };
                        }
                    }
                }
                // Tensor element-wise arithmetic
                if let (
                    Ty::Tensor {
                        elem: ea,
                        shape: sa,
                    },
                    Ty::Tensor {
                        elem: eb,
                        shape: sb,
                    },
                ) = (&l, &r)
                {
                    if ea != eb {
                        self.diag.error(
                            span,
                            format!(
                                "tensor element type mismatch: `{}` vs `{}`",
                                l.display(),
                                r.display()
                            ),
                        );
                    } else if sa.len() != sb.len() {
                        self.diag.error(
                            span,
                            format!(
                                "tensor rank mismatch: rank {} vs rank {}",
                                sa.len(),
                                sb.len()
                            ),
                        );
                    }
                    return l;
                }
                // Tensor + scalar broadcast
                if let (Ty::Tensor { .. }, ref scalar) | (ref scalar, Ty::Tensor { .. }) = (&l, &r)
                {
                    if scalar.is_numeric() {
                        return if matches!(&l, Ty::Tensor { .. }) {
                            l.clone()
                        } else {
                            r.clone()
                        };
                    }
                }
                // Plain numeric
                if !self.infer.unify(&l, &r) {
                    self.diag.error(
                        span,
                        format!(
                            "type mismatch in arithmetic: `{}` {} `{}`",
                            l.display(),
                            format!("{:?}", op).to_lowercase(),
                            r.display()
                        ),
                    );
                }
                l
            }

            // Bitwise
            BinOpKind::BitAnd
            | BinOpKind::BitOr
            | BinOpKind::BitXor
            | BinOpKind::Shl
            | BinOpKind::Shr => {
                if !l.is_numeric() || !r.is_numeric() {
                    self.diag
                        .error(span, "bitwise operators require integer operands");
                }
                l
            }
        }
    }

    // =========================================================================
    // TENSOR ALGEBRA CHECKING (Feature 1)
    // =========================================================================

    /// `A @ B` — matrix multiply.
    /// Rules:
    ///   tensor<E>[M, K] @ tensor<E>[K, N] → tensor<E>[M, N]
    ///   mat3 @ mat3 → mat3
    ///   mat3 @ vec3 → vec3    (M×N × N → M)
    fn check_matmul(&mut self, span: Span, lhs: &Expr, rhs: &Expr, env: &mut TyEnv) -> Ty {
        let l = self.check_expr(lhs, env);
        let r = self.check_expr(rhs, env);

        // mat × mat
        if let (Ty::Mat { size: ls }, Ty::Mat { size: rs }) = (&l, &r) {
            if ls != rs {
                self.diag.error(
                    span,
                    format!(
                        "matrix multiply: size mismatch `{}` @ `{}`",
                        l.display(),
                        r.display()
                    ),
                );
            }
            return Ty::Mat { size: *ls };
        }

        // mat × vec
        if let (Ty::Mat { size: ms }, Ty::Vec { size: vs, .. }) = (&l, &r) {
            if ms != vs {
                self.diag.error(
                    span,
                    format!(
                        "matrix–vector multiply: size mismatch `{}` @ `{}`",
                        l.display(),
                        r.display()
                    ),
                );
            }
            return Ty::Vec {
                size: *vs,
                family: VecFamily::Float,
            };
        }

        // tensor @ tensor
        if let (
            Ty::Tensor {
                elem: ea,
                shape: sa,
            },
            Ty::Tensor {
                elem: eb,
                shape: sb,
            },
        ) = (&l, &r)
        {
            if ea != eb {
                self.diag.error(
                    span,
                    format!(
                        "matrix multiply: element type mismatch \
                     (`tensor<{:?}>` @ `tensor<{:?}>`)",
                        ea, eb
                    ),
                );
            }
            // Require at least 2-D tensors.
            if sa.len() < 2 || sb.len() < 2 {
                self.diag
                    .error(span, "matrix multiply (`@`) requires at least 2-D tensors");
                return self.infer.fresh();
            }
            // Inner dimensions must agree.
            let k_l = &sa[sa.len() - 1];
            let k_r = &sb[sb.len() - 2];
            if let (Dim::Lit(kl), Dim::Lit(kr)) = (k_l, k_r) {
                if kl != kr {
                    self.diag.error(
                        span,
                        format!(
                            "matrix multiply shape mismatch: \
                         inner dim {} (lhs) ≠ {} (rhs)",
                            kl, kr
                        ),
                    );
                }
            }
            // Result shape: all but last dim of lhs ++ all but second-to-last of rhs.
            let mut result_shape: Vec<Dim> = sa[..sa.len() - 1].to_vec();
            result_shape.push(sb[sb.len() - 1].clone());
            return Ty::Tensor {
                elem: ea.clone(),
                shape: result_shape,
            };
        }

        self.diag.error(
            span,
            format!(
                "`@` (matrix multiply) requires tensor or matrix operands; \
             got `{}` and `{}`",
                l.display(),
                r.display()
            ),
        );
        self.infer.fresh()
    }

    /// `A .* B` or `A ./ B` — element-wise (Hadamard) multiply/divide.
    /// Supports numpy-style broadcasting: [3,1] .* [1,4] → [3,4].
    fn check_hadamard(&mut self, span: Span, lhs: &Expr, rhs: &Expr, env: &mut TyEnv) -> Ty {
        let l = self.check_expr(lhs, env);
        let r = self.check_expr(rhs, env);

        if !l.supports_hadamard() || !r.supports_hadamard() {
            self.diag.error(
                span,
                format!(
                    "Hadamard operator (`.*`/`./`) requires tensor or vector operands; \
                 got `{}` and `{}`",
                    l.display(),
                    r.display()
                ),
            );
            return self.infer.fresh();
        }

        // Numpy-style broadcasting for tensors
        if let (
            Ty::Tensor {
                elem: ea,
                shape: sa,
            },
            Ty::Tensor {
                elem: eb,
                shape: sb,
            },
        ) = (&l, &r)
        {
            if ea != eb {
                self.diag.error(
                    span,
                    format!(
                        "Hadamard operator: element type mismatch `{}` vs `{}`",
                        l.display(),
                        r.display()
                    ),
                );
                return self.infer.fresh();
            }
            // Broadcast: align from the right, check compatibility
            let result_shape = broadcast_shapes(sa, sb);
            match result_shape {
                Some(shape) => {
                    return Ty::Tensor {
                        elem: ea.clone(),
                        shape,
                    }
                }
                None => {
                    self.diag.error(
                        span,
                        format!("cannot broadcast tensor shapes {:?} and {:?}", sa, sb),
                    );
                    return self.infer.fresh();
                }
            }
        }

        if !self.infer.unify(&l, &r) {
            self.diag.error(
                span,
                format!(
                    "Hadamard operator: operand type mismatch `{}` vs `{}`",
                    l.display(),
                    r.display()
                ),
            );
        }
        l
    }

    /// `A ++ B` — tensor concatenation along axis 0.
    /// Both operands must be tensors of the same rank and element type.
    /// All dims except axis 0 must match.
    fn check_tensor_concat(&mut self, span: Span, lhs: &Expr, rhs: &Expr, env: &mut TyEnv) -> Ty {
        let l = self.check_expr(lhs, env);
        let r = self.check_expr(rhs, env);

        if let (
            Ty::Tensor {
                elem: ea,
                shape: sa,
            },
            Ty::Tensor {
                elem: eb,
                shape: sb,
            },
        ) = (&l, &r)
        {
            if ea != eb {
                self.diag.error(
                    span,
                    format!(
                        "tensor concat `++`: element type mismatch \
                     (`tensor<{:?}>` ++ `tensor<{:?}>`)",
                        ea, eb
                    ),
                );
            }
            if sa.len() != sb.len() {
                self.diag.error(
                    span,
                    format!(
                        "tensor concat `++`: rank mismatch ({} vs {})",
                        sa.len(),
                        sb.len()
                    ),
                );
                return self.infer.fresh();
            }
            // All dims except axis 0 must be compatible.
            for (i, (da, db)) in sa[1..].iter().zip(sb[1..].iter()).enumerate() {
                if let (Dim::Lit(a), Dim::Lit(b)) = (da, db) {
                    if a != b {
                        self.diag.error(
                            span,
                            format!(
                                "tensor concat `++`: dimension {} mismatch ({} vs {})",
                                i + 1,
                                a,
                                b
                            ),
                        );
                    }
                }
            }
            // Result: axis-0 is the sum (or Dynamic if either is Dynamic/Named).
            let new_d0 = match (&sa[0], &sb[0]) {
                (Dim::Lit(a), Dim::Lit(b)) => Dim::Lit(a + b),
                _ => Dim::Dynamic,
            };
            let mut result_shape = vec![new_d0];
            result_shape.extend(sa[1..].iter().cloned());
            return Ty::Tensor {
                elem: ea.clone(),
                shape: result_shape,
            };
        }

        self.diag.error(
            span,
            format!(
                "`++` (tensor concat) requires tensor operands; got `{}` and `{}`",
                l.display(),
                r.display()
            ),
        );
        self.infer.fresh()
    }

    // =========================================================================
    // FIELD ACCESS CHECKING
    // =========================================================================

    fn check_field_access(
        &mut self,
        span: Span,
        object: &Expr,
        field: &str,
        env: &mut TyEnv,
    ) -> Ty {
        let obj_ty = self.check_expr(object, env);
        let resolved = self.infer.resolve(&obj_ty);

        match &resolved {
            // Struct or component field lookup.
            Ty::Struct(name) | Ty::Component(name) => {
                let name = name.clone();
                if let Some(info) = self.symbols.structs.get(&name).cloned() {
                    if let Some(ty) = info.field_ty(field) {
                        ty.clone()
                    } else {
                        self.diag
                            .error(span, format!("`{}` has no field `{}`", name, field));
                        self.infer.fresh()
                    }
                } else {
                    self.infer.fresh()
                }
            }

            // vec2/3/4 swizzle components.
            Ty::Vec { size, family } => {
                let lanes = size.lanes() as usize;
                let valid: &[char] = match lanes {
                    2 => &['x', 'y'],
                    3 => &['x', 'y', 'z'],
                    4 => &['x', 'y', 'z', 'w'],
                    _ => &[],
                };
                // Swizzle: any combo of valid components.
                if field.chars().all(|c| valid.contains(&c)) {
                    match field.len() {
                        1 => Ty::Scalar(ElemType::F32), // approximate
                        2 => Ty::Vec {
                            size: VecSize::N2,
                            family: *family,
                        },
                        3 => Ty::Vec {
                            size: VecSize::N3,
                            family: *family,
                        },
                        4 => Ty::Vec {
                            size: VecSize::N4,
                            family: *family,
                        },
                        _ => {
                            self.diag.error(span, "swizzle exceeds 4 components");
                            self.infer.fresh()
                        }
                    }
                } else {
                    self.diag.error(
                        span,
                        format!("`{}` has no field `{}`", resolved.display(), field),
                    );
                    self.infer.fresh()
                }
            }

            // Quat components.
            Ty::Quat => {
                if matches!(field, "x" | "y" | "z" | "w") {
                    Ty::Scalar(ElemType::F32)
                } else {
                    self.diag
                        .error(span, format!("`quat` has no field `{}`", field));
                    self.infer.fresh()
                }
            }

            // Tuple index: "0", "1", …
            Ty::Tuple(ts) => {
                if let Ok(idx) = field.parse::<usize>() {
                    if idx < ts.len() {
                        ts[idx].clone()
                    } else {
                        self.diag.error(
                            span,
                            format!("tuple index {} out of range (len {})", idx, ts.len()),
                        );
                        self.infer.fresh()
                    }
                } else {
                    self.diag
                        .error(span, format!("invalid tuple field `{}`", field));
                    self.infer.fresh()
                }
            }

            _ => {
                self.diag.error(
                    span,
                    format!("type `{}` has no field `{}`", resolved.display(), field),
                );
                self.infer.fresh()
            }
        }
    }

    // =========================================================================
    // CALL CHECKING
    // =========================================================================

    fn check_call(
        &mut self,
        span: Span,
        func: &Expr,
        args: &[Expr],
        _named: &[(String, Expr)],
        env: &mut TyEnv,
    ) -> Ty {
        // Resolve callee type.
        let func_ty = self.check_expr(func, env);

        // Look up argument types.
        let arg_tys: Vec<Ty> = args.iter().map(|a| self.check_expr(a, env)).collect();

        match &func_ty {
            Ty::FnPtr { params, ret } => {
                if params.len() != arg_tys.len() {
                    self.diag.error(
                        span,
                        format!(
                            "function expects {} argument(s), {} provided",
                            params.len(),
                            arg_tys.len()
                        ),
                    );
                }
                for (i, (expected, actual)) in params.iter().zip(arg_tys.iter()).enumerate() {
                    if !self.infer.unify(expected, actual) {
                        self.diag.error(
                            span,
                            format!(
                                "argument {} type mismatch: expected `{}`, got `{}`",
                                i + 1,
                                expected.display(),
                                actual.display()
                            ),
                        );
                    }
                }
                *ret.clone()
            }
            // Try to resolve via the global function table when the callee is an Ident.
            _ => {
                if let Expr::Ident { name, .. } = func {
                    if let Some(fi) = self.symbols.fns.get(name.as_str()).cloned() {
                        if fi.params.len() != arg_tys.len() {
                            self.diag.error(
                                span,
                                format!(
                                    "function `{}` expects {} argument(s), {} provided",
                                    name,
                                    fi.params.len(),
                                    arg_tys.len()
                                ),
                            );
                        }
                        for (i, ((_, expected), actual)) in
                            fi.params.iter().zip(arg_tys.iter()).enumerate()
                        {
                            if !self.infer.unify(expected, actual) {
                                self.diag.error(
                                    span,
                                    format!(
                                        "argument {} to `{}`: expected `{}`, got `{}`",
                                        i + 1,
                                        name,
                                        expected.display(),
                                        actual.display()
                                    ),
                                );
                            }
                        }
                        return fi.ret.clone();
                    }
                    // Unknown function — emit error, return fresh var.
                    self.diag
                        .error(span, format!("call to undeclared function `{}`", name));
                } else {
                    self.diag.error(
                        span,
                        format!("expression of type `{}` is not callable", func_ty.display()),
                    );
                }
                self.infer.fresh()
            }
        }
    }

    fn check_method_call(
        &mut self,
        span: Span,
        receiver: &Expr,
        method: &str,
        args: &[Expr],
        env: &mut TyEnv,
    ) -> Ty {
        let recv_ty = self.check_expr(receiver, env);
        let _arg_tys: Vec<Ty> = args.iter().map(|a| self.check_expr(a, env)).collect();

        // Built-in tensor / vector methods.
        match (&recv_ty, method) {
            (Ty::Tensor { elem, shape }, "transpose") => {
                if shape.len() < 2 {
                    self.diag.error(span, "`transpose()` requires a 2-D tensor");
                    return recv_ty;
                }
                let mut new_shape = shape.clone();
                let n = new_shape.len();
                new_shape.swap(n - 2, n - 1);
                Ty::Tensor {
                    elem: elem.clone(),
                    shape: new_shape,
                }
            }
            (Ty::Tensor { elem, .. }, "reshape") => {
                // Return a dynamic tensor of the same element type.
                Ty::Tensor {
                    elem: elem.clone(),
                    shape: vec![Dim::Dynamic],
                }
            }
            (Ty::Tensor { elem, .. }, "sum" | "mean" | "max" | "min") => Ty::Scalar(elem.clone()),
            (Ty::Tensor { elem, shape }, "softmax" | "relu" | "sigmoid" | "tanh") => Ty::Tensor {
                elem: elem.clone(),
                shape: shape.clone(),
            },
            (Ty::Vec { .. }, "dot") => Ty::Scalar(ElemType::F32),
            (Ty::Vec { .. }, "normalize") => recv_ty.clone(),
            (Ty::Vec { .. }, "length" | "magnitude") => Ty::Scalar(ElemType::F32),
            (
                Ty::Vec {
                    size: VecSize::N3, ..
                },
                "cross",
            ) => recv_ty.clone(),
            (Ty::Quat, "normalize" | "conjugate" | "inverse") => Ty::Quat,
            (Ty::Quat, "to_mat3") => Ty::Mat { size: VecSize::N3 },
            (Ty::Quat, "to_mat4") => Ty::Mat { size: VecSize::N4 },

            _ => {
                // Unknown method — defer gracefully.
                self.diag.note(
                    span,
                    format!(
                        "unknown method `{}` on type `{}`; \
                     type inference result may be incomplete",
                        method,
                        recv_ty.display()
                    ),
                );
                self.infer.fresh()
            }
        }
    }

    // =========================================================================
    // BLOCK / STATEMENT CHECKING
    // =========================================================================

    /// Check a block.  Returns the type of the trailing expression, or `Unit`.
    pub fn check_block(&mut self, block: &Block, env: &mut TyEnv) -> Ty {
        env.push_scope();
        for stmt in &block.stmts {
            self.check_stmt(stmt, env);
        }
        let ty = if let Some(tail) = &block.tail {
            self.check_expr(tail, env)
        } else {
            Ty::Unit
        };
        env.pop_scope();
        ty
    }

    fn check_stmt(&mut self, stmt: &crate::ast::Stmt, env: &mut TyEnv) {
        use crate::ast::Stmt;
        match stmt {
            Stmt::Let {
                span,
                pattern,
                ty,
                init,
                ..
            } => {
                let declared = ty
                    .as_ref()
                    .map(|t| self.lower_ast_type(t, *span))
                    .unwrap_or_else(|| self.infer.fresh());

                let init_ty = init
                    .as_ref()
                    .map(|e| self.check_expr(e, env))
                    .unwrap_or_else(|| self.infer.fresh());

                if !self.infer.unify(&declared, &init_ty) {
                    self.diag.error(
                        *span,
                        format!(
                            "type mismatch in `let`: declared `{}`, initialiser is `{}`",
                            declared.display(),
                            init_ty.display()
                        ),
                    );
                }

                // Bind pattern names.
                self.bind_pattern(pattern, &declared, env);
            }

            Stmt::Expr { expr, .. } => {
                self.check_expr(expr, env);
            }

            Stmt::Return { span, value } => {
                if let Some(e) = value {
                    self.check_expr(e, env);
                } else {
                    let _ = *span;
                }
            }

            Stmt::ForIn {
                span,
                pattern,
                iter,
                body,
                ..
            } => {
                let iter_ty = self.check_expr(iter, env);
                let elem_ty = match &iter_ty {
                    Ty::Array { elem, .. } | Ty::Slice(elem) => *elem.clone(),
                    Ty::Tensor { elem, shape } if shape.len() == 1 => Ty::Scalar(elem.clone()),
                    _ => self.infer.fresh(),
                };
                env.push_scope();
                self.bind_pattern(pattern, &elem_ty, env);
                self.check_block(body, env);
                env.pop_scope();
                let _ = *span;
            }

            Stmt::EntityFor {
                span,
                var,
                query,
                body,
                ..
            } => {
                // Validate component names in the query.
                for comp in query.with.iter().chain(query.without.iter()) {
                    if !self.symbols.structs.contains_key(comp.as_str()) {
                        self.diag.error(
                            *span,
                            format!("entity query references unknown component `{}`", comp),
                        );
                    }
                }
                if let Some(f) = &query.filter {
                    let ft = self.check_expr(f, env);
                    if !matches!(ft, Ty::Bool) {
                        self.diag
                            .error(*span, "entity query filter must be a `bool` expression");
                    }
                }
                env.push_scope();
                // The loop variable is an opaque entity handle.
                env.bind(var.clone(), Ty::Struct("Entity".into()));
                self.check_block(body, env);
                env.pop_scope();
            }

            Stmt::While {
                span: _,
                cond,
                body,
                ..
            } => {
                let ct = self.check_expr(cond, env);
                if !matches!(ct, Ty::Bool) {
                    self.diag.error(
                        cond.span(),
                        format!("`while` condition must be `bool`, got `{}`", ct.display()),
                    );
                }
                self.check_block(body, env);
            }

            Stmt::Loop { body, .. } => {
                self.check_block(body, env);
            }

            Stmt::If {
                span,
                cond,
                then,
                else_,
                ..
            } => {
                let ct = self.check_expr(cond, env);
                if !matches!(ct, Ty::Bool) {
                    self.diag.error(
                        *span,
                        format!("`if` condition must be `bool`, got `{}`", ct.display()),
                    );
                }
                self.check_block(then, env);
                if let Some(e) = else_ {
                    match e.as_ref() {
                        crate::ast::IfOrBlock::If(s) => self.check_stmt(s, env),
                        crate::ast::IfOrBlock::Block(b) => {
                            self.check_block(b, env);
                        }
                    }
                }
            }

            Stmt::Match {
                span: _,
                expr,
                arms,
            } => {
                let scrutinee = self.check_expr(expr, env);
                let mut arm_ty: Option<Ty> = None;
                for arm in arms {
                    env.push_scope();
                    self.bind_pattern(&arm.pat, &scrutinee, env);
                    if let Some(g) = &arm.guard {
                        let gt = self.check_expr(g, env);
                        if !matches!(gt, Ty::Bool) {
                            self.diag.error(arm.span, "match guard must be `bool`");
                        }
                    }
                    let bt = self.check_expr(&arm.body, env);
                    if let Some(ref prev) = arm_ty {
                        if !self.infer.unify(prev, &bt) {
                            self.diag.error(
                                arm.span,
                                format!(
                                    "match arm type `{}` is incompatible with `{}`",
                                    bt.display(),
                                    prev.display()
                                ),
                            );
                        }
                    } else {
                        arm_ty = Some(bt);
                    }
                    env.pop_scope();
                }
            }

            Stmt::Item(i) => self.check_item(i),

            Stmt::Break { value, .. } => {
                if let Some(e) = value {
                    self.check_expr(e, env);
                }
            }
            Stmt::Continue { .. } => {}

            Stmt::ParallelFor(pf) => {
                let iter_ty = self.check_expr(&pf.iter, env);
                let elem_ty = match &iter_ty {
                    Ty::Array { elem, .. } | Ty::Slice(elem) => *elem.clone(),
                    _ => self.infer.fresh(),
                };
                env.push_scope();
                self.bind_pattern(&pf.var, &elem_ty, env);
                self.check_block(&pf.body, env);
                env.pop_scope();
            }

            Stmt::Spawn(sb) => {
                self.check_block(&sb.body, env);
            }
            Stmt::Sync(sb) => {
                self.check_block(&sb.body, env);
            }
            Stmt::Atomic(ab) => {
                self.check_block(&ab.body, env);
            }
        }
    }

    // =========================================================================
    // PATTERN BINDING
    // =========================================================================

    fn bind_pattern(&mut self, pat: &Pattern, ty: &Ty, env: &mut TyEnv) {
        match pat {
            Pattern::Ident { name, .. } => env.bind(name.clone(), ty.clone()),
            Pattern::Wildcard(_) => {}
            Pattern::Tuple { elems, span } => {
                if let Ty::Tuple(ts) = ty {
                    if ts.len() != elems.len() {
                        self.diag.error(
                            *span,
                            format!(
                                "tuple pattern has {} elements but type has {}",
                                elems.len(),
                                ts.len()
                            ),
                        );
                    }
                    for (p, t) in elems.iter().zip(ts.iter()) {
                        self.bind_pattern(p, t, env);
                    }
                } else {
                    for p in elems {
                        let fresh_ty = self.infer.fresh();
                        self.bind_pattern(p, &fresh_ty, env);
                    }
                }
            }
            Pattern::Struct { fields, .. } => {
                for (fname, maybe_pat) in fields {
                    let fty = if let Ty::Struct(name) | Ty::Component(name) = ty {
                        self.symbols
                            .structs
                            .get(name)
                            .and_then(|s| s.field_ty(fname))
                            .cloned()
                            .unwrap_or_else(|| self.infer.fresh())
                    } else {
                        self.infer.fresh()
                    };
                    if let Some(p) = maybe_pat {
                        self.bind_pattern(p, &fty, env);
                    } else {
                        env.bind(fname.clone(), fty);
                    }
                }
            }
            Pattern::Lit(..)
            | Pattern::Range { .. }
            | Pattern::Or { .. }
            | Pattern::Enum { .. } => {}
        }
    }

    // =========================================================================
    // HELPER — type expectation with a helpful error message
    // =========================================================================

    fn expect_ty(&mut self, expected: &Ty, actual: &Ty, span: Span, ctx: &str) {
        if !self.infer.unify(expected, actual) {
            self.diag.error(
                span,
                format!(
                    "type mismatch in {}: expected `{}`, got `{}`",
                    ctx,
                    expected.display(),
                    actual.display()
                ),
            );
        }
    }

    // =========================================================================
    // NEURAL-NETWORK MODEL CHECKING (Unique Feature 1)
    // =========================================================================

    fn check_model(&mut self, m: &ModelDecl) {
        // Validate dropout rates.
        for layer in &m.layers {
            if let ModelLayer::Dropout { span, rate } = layer {
                if *rate < 0.0 || *rate >= 1.0 {
                    self.diag.error(
                        *span,
                        format!("dropout rate {} is out of range [0.0, 1.0)", rate),
                    );
                }
            }
            if let ModelLayer::Attention {
                span,
                num_heads,
                head_dim,
            } = layer
            {
                if *num_heads == 0 || *head_dim == 0 {
                    self.diag.error(
                        *span,
                        "attention layer must have num_heads > 0 and head_dim > 0",
                    );
                }
            }
            if let ModelLayer::Embed {
                span,
                vocab_size,
                embed_dim,
            } = layer
            {
                if *vocab_size == 0 || *embed_dim == 0 {
                    self.diag.error(
                        *span,
                        "embed layer must have vocab_size > 0 and embed_dim > 0",
                    );
                }
            }
        }

        // Validate layer connectivity: track the "current output width" where known.
        let mut current_width: Option<u64> = None;
        for layer in &m.layers {
            match layer {
                ModelLayer::Input { size, .. } => {
                    current_width = Some(*size);
                }
                ModelLayer::Dense { span, units, .. } => {
                    current_width = Some(*units);
                    if *units == 0 {
                        self.diag.error(*span, "dense layer must have units > 0");
                    }
                }
                ModelLayer::Output { span, units, .. } => {
                    if *units == 0 {
                        self.diag.error(*span, "output layer must have units > 0");
                    }
                    current_width = Some(*units);
                }
                ModelLayer::Pool {
                    span,
                    size_h,
                    size_w,
                    ..
                } => {
                    if *size_h == 0 || *size_w == 0 {
                        self.diag.error(*span, "pool size must be > 0");
                    }
                }
                ModelLayer::Conv2d {
                    span,
                    filters,
                    kernel_h,
                    kernel_w,
                    ..
                } => {
                    if *filters == 0 || *kernel_h == 0 || *kernel_w == 0 {
                        self.diag.error(
                            *span,
                            "conv layer must have filters > 0 and kernel size > 0",
                        );
                    }
                }
                ModelLayer::SubModel { span, name } => {
                    if !self.symbols.models.contains_key(name.as_str()) {
                        self.diag.error(
                            *span,
                            format!("model layer references unknown model `{}`", name),
                        );
                    }
                }
                _ => {}
            }
            let _ = current_width; // suppress unused warning for now
        }

        // Warn if trainable model has no @grad attribute.
        if !m.is_trainable() && m.optimizer.is_some() {
            self.diag.warning(
                m.span,
                "model has an optimizer but no `@grad` attribute; \
                 gradients will not be tracked",
            );
        }
    }

    // =========================================================================
    // TRAINING DECLARATION CHECKING (Unique Feature 2)
    // =========================================================================

    fn check_train(&mut self, t: &TrainDecl) {
        // Verify the agent exists.
        if !self.symbols.agents.contains_key(&t.agent) {
            self.diag.error(
                t.span,
                format!("`train` references unknown agent `{}`", t.agent),
            );
        }

        // Verify the model exists (if specified).
        if let Some(model_name) = &t.model {
            if !self.symbols.models.contains_key(model_name.as_str()) {
                self.diag.error(
                    t.span,
                    format!("`train` references unknown model `{}`", model_name),
                );
            }
        }

        // Signal weights should be non-negative.
        for sig in &t.signals {
            if sig.weight < 0.0 {
                self.diag.error(
                    sig.span,
                    format!(
                        "{} signal `{}` has negative weight {}; \
                     use a `penalty` with a positive weight for negative reinforcement",
                        if sig.is_reward { "reward" } else { "penalty" },
                        sig.name,
                        sig.weight
                    ),
                );
            }
            // Check expression if present.
            if let Some(e) = &sig.expr {
                let mut env = TyEnv::new();
                let ty = self.check_expr(e, &mut env);
                if !ty.is_float_scalar() {
                    self.diag.warning(
                        e.span(),
                        format!(
                            "signal expression for `{}` should produce a scalar float; \
                         got `{}`",
                            sig.name,
                            ty.display()
                        ),
                    );
                }
            }
        }

        // Check episode done-condition.
        if let Some(ep) = &t.episode {
            if let Some(cond) = &ep.done_condition {
                let mut env = TyEnv::new();
                let ty = self.check_expr(cond, &mut env);
                if !matches!(ty, Ty::Bool) {
                    self.diag.error(
                        cond.span(),
                        format!(
                            "episode `done_condition` must be `bool`, got `{}`",
                            ty.display()
                        ),
                    );
                }
            }
        }

        // Check hyper-parameter expressions.
        let mut env = TyEnv::new();
        for (name, expr) in &t.hyper {
            let ty = self.check_expr(expr, &mut env);
            if !ty.is_float_scalar() && !matches!(ty, Ty::Scalar(_)) {
                self.diag.warning(
                    expr.span(),
                    format!(
                        "hyper-parameter `{}` value should be a scalar; got `{}`",
                        name,
                        ty.display()
                    ),
                );
            }
        }
    }

    // =========================================================================
    // AGENT DECLARATION CHECKING (Feature 3)
    // =========================================================================

    fn check_agent(&mut self, a: &AgentDecl) {
        // Validate network architecture decorators like:
        // @AI("128->64->32"), @PPO(128->64->32), @DQN(256->128->64)
        for attr in &a.attrs {
            if let crate::ast::Attribute::Named { name, .. } = attr {
                if is_network_decorator(name) {
                    self.validate_network_decorator(a, attr, name);
                }
            }
        }

        // If learning is reinforcement/imitation, there should be a policy model.
        if let Some(ls) = &a.learning {
            match &ls.kind {
                crate::ast::LearningKind::Reinforcement | crate::ast::LearningKind::Imitation => {
                    if ls.policy_model.is_none() {
                        self.diag.warning(
                            ls.span,
                            format!(
                                "agent `{}` uses {:?} learning but has no `policy_model`; \
                             a model must be referenced in a `train` block",
                                a.name, ls.kind
                            ),
                        );
                    } else if let Some(pm) = &ls.policy_model {
                        if !self.symbols.models.contains_key(pm.as_str()) {
                            self.diag.error(
                                ls.span,
                                format!(
                                    "agent `{}` references unknown policy model `{}`",
                                    a.name, pm
                                ),
                            );
                        }
                    }
                    if let Some(lr) = ls.learning_rate {
                        if lr <= 0.0 {
                            self.diag
                                .error(ls.span, format!("learning rate must be > 0, got {}", lr));
                        }
                    }
                    if let Some(g) = ls.gamma {
                        if !(0.0..=1.0).contains(&g) {
                            self.diag.error(
                                ls.span,
                                format!("discount factor γ must be in [0, 1], got {}", g),
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        // Check behaviour rule bodies.
        for rule in &a.behaviors {
            let mut env = TyEnv::new();
            env.push_scope();
            // Bind `self` as the agent type.
            env.bind("self", Ty::Struct(a.name.clone()));
            for param in &rule.params {
                let ty = param
                    .ty
                    .as_ref()
                    .map(|t| self.lower_ast_type(t, param.span))
                    .unwrap_or_else(|| self.infer.fresh());
                env.bind(param.name.clone(), ty);
            }
            self.check_block(&rule.body, &mut env);
            env.pop_scope();
        }

        // Check goal utility expressions: must be scalar float in [0, 1].
        for goal in &a.goals {
            let mut env = TyEnv::new();
            env.bind("self", Ty::Struct(a.name.clone()));
            let ty = self.check_expr(&goal.utility, &mut env);
            if !ty.is_float_scalar() {
                self.diag.error(
                    goal.utility.span(),
                    format!(
                        "goal `{}` utility expression should return a float scalar \
                     (representing priority in [0, 1]); got `{}`",
                        goal.name,
                        ty.display()
                    ),
                );
            }
        }

        // Warn if there are no behaviours.
        if a.behaviors.is_empty() {
            self.diag
                .warning(a.span, format!("agent `{}` declares no behaviours", a.name));
        }
    }

    // ─── Network Decorator Validation ──────────────────────────────────────────
    fn validate_network_decorator(
        &mut self,
        a: &AgentDecl,
        attr: &crate::ast::Attribute,
        decorator_name: &str,
    ) {
        let Some(cfg) = self.extract_ai_decorator_config(a, attr, decorator_name) else { return };

        if let Err(e) = self.validate_architecture_string(&cfg.network) {
            self.diag
                .error(cfg.span, format!("@{}: {}", decorator_name.to_uppercase(), e));
            return;
        }

        let layers = match self.extract_layers_from_arch(&cfg.network) {
            Ok(layers) => layers,
            Err(_) => {
                self.diag.error(
                    cfg.span,
                    format!("@{} architecture format is invalid", decorator_name.to_uppercase()),
                );
                return;
            }
        };

        // Calculate total perception size
        let total_perception_size: u64 = a
            .perceptions
            .iter()
            .map(|p| match &p.range {
                Some(r) => *r as u64,
                None => 1,
            })
            .sum();

        // Check if input layer matches perception size
        if let Some(input_size) = layers.first() {
            if *input_size != total_perception_size {
                self.diag.error(cfg.span, format!(
                    "@AI input layer size {} doesn't match total perception size {}. \
                     Perceptions: {}",
                    *input_size,
                    total_perception_size,
                    a.perceptions.iter()
                        .map(|p| format!("{} ({})", p.tag.as_deref().unwrap_or("perception"), p.range.unwrap_or(1.0)))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }

        if let Some(input) = cfg.input {
            if let Some(first) = layers.first() {
                if *first != input {
                    self.diag.error(
                        cfg.span,
                        format!(
                            "@AI(input={}) does not match network input layer {}",
                            input, first
                        ),
                    );
                }
            }
        }

        if let Some(output) = cfg.output {
            if let Some(last) = layers.last() {
                if *last != output {
                    self.diag.error(
                        cfg.span,
                        format!(
                            "@AI(output={}) does not match network output layer {}",
                            output, last
                        ),
                    );
                }
            }
        }

        if let Some(lr) = cfg.learning_rate {
            if !lr.is_finite() || lr <= 0.0 {
                self.diag
                    .error(cfg.span, format!("@AI learning rate must be > 0, got {}", lr));
            }
        }
    }

    fn extract_ai_decorator_config(
        &mut self,
        a: &AgentDecl,
        attr: &crate::ast::Attribute,
        decorator_name: &str,
    ) -> Option<AiDecoratorConfig> {
        let crate::ast::Attribute::Named { args, .. } = attr else { return None };
        let mut network: Option<String> = None;
        let mut learning_rate: Option<f64> = None;
        let mut input: Option<u64> = None;
        let mut output: Option<u64> = None;
        let mut span = a.span;

        for (idx, arg) in args.iter().enumerate() {
            span = arg.span();
            match arg {
                crate::ast::Expr::StrLit { value, .. } if idx == 0 => {
                    network = Some(value.clone());
                }
                crate::ast::Expr::Assign { target, value, .. } => {
                    let key = match target.as_ref() {
                        crate::ast::Expr::Ident { name, .. } => name.as_str(),
                        crate::ast::Expr::Path { segments, .. } if segments.len() == 1 => {
                            segments[0].as_str()
                        }
                        _ => {
                            self.diag.error(
                                target.span(),
                                "@AI options must use simple keys like `network=...`, `lr=...`, `input=...`, `output=...`",
                            );
                            continue;
                        }
                    };

                    match key {
                        "network" | "arch" | "architecture" => {
                            if let crate::ast::Expr::StrLit { value, .. } = value.as_ref() {
                                network = Some(value.clone());
                            } else {
                                self.diag.error(value.span(), "@AI `network` must be a string, e.g. network=\"128->64->10\"");
                            }
                        }
                        "lr" | "learning_rate" => {
                            if let Some(v) = self.ai_number_literal(value) {
                                learning_rate = Some(v);
                            } else {
                                self.diag.error(value.span(), "@AI `lr`/`learning_rate` must be numeric");
                            }
                        }
                        "input" => {
                            if let Some(v) = self.ai_u64_literal(value) {
                                input = Some(v);
                            } else {
                                self.diag.error(value.span(), "@AI `input` must be an integer");
                            }
                        }
                        "output" => {
                            if let Some(v) = self.ai_u64_literal(value) {
                                output = Some(v);
                            } else {
                                self.diag.error(value.span(), "@AI `output` must be an integer");
                            }
                        }
                        other => {
                            self.diag.warning(
                                target.span(),
                                format!(
                                    "@AI unknown option `{}`; supported: network, lr, input, output",
                                    other
                                ),
                            );
                        }
                    }
                }
                crate::ast::Expr::StrLit { value, .. } => {
                    network = Some(value.clone());
                }
                _ => {}
            }
        }

        let Some(network) = network else {
            self.diag.error(
                a.span,
                format!(
                    "@{} requires a network. Use @{}(\"128->64->10\") or @{}(network=\"128->64->10\", lr=0.0003, input=128, output=10)",
                    decorator_name.to_uppercase(),
                    decorator_name.to_uppercase(),
                    decorator_name.to_uppercase()
                ),
            );
            return None;
        };

        Some(AiDecoratorConfig {
            span,
            network,
            learning_rate,
            input,
            output,
        })
    }

    fn ai_number_literal(&self, e: &crate::ast::Expr) -> Option<f64> {
        match e {
            crate::ast::Expr::FloatLit { value, .. } => Some(*value),
            crate::ast::Expr::IntLit { value, .. } => Some(*value as f64),
            _ => None,
        }
    }

    fn ai_u64_literal(&self, e: &crate::ast::Expr) -> Option<u64> {
        match e {
            crate::ast::Expr::IntLit { value, .. } => Some(*value as u64),
            _ => None,
        }
    }

    /// Validate that an architecture string has valid format
    fn validate_architecture_string(&self, s: &str) -> Result<(), String> {
        let s = s.trim();
        if s.is_empty() {
            return Err("empty architecture specification".into());
        }

        // Check for type prefix
        let (arch_type, rest) = if let Some(colon_pos) = s.find(':') {
            let (prefix, tail) = s.split_at(colon_pos);
            (prefix.to_lowercase(), &tail[1..]) // Skip the ':'
        } else {
            ("mlp".into(), s)
        };

        // Validate architecture type
        match arch_type.as_str() {
            "mlp" | "lstm" | "cnn" | "dueling" | "transformer" => {}
            _ => {
                return Err(format!(
                "unknown architecture type `{}`. Supported: mlp, lstm, cnn, dueling, transformer",
                arch_type
            ))
            }
        }

        // Basic format validation: should have numbers and operators
        if rest.is_empty() {
            return Err(format!("architecture `{}:` is incomplete", arch_type));
        }

        // For MLP/LSTM: should have -> separators
        if matches!(arch_type.as_str(), "mlp" | "lstm") && !rest.contains("->") {
            return Err(format!(
                "`{}` architecture must have layers separated by `->` (e.g., `{}:256->512->10`)",
                arch_type, arch_type
            ));
        }

        Ok(())
    }

    /// Extract layer dimensions from architecture string (MLP format for now)
    fn extract_layers_from_arch(&self, s: &str) -> Result<Vec<u64>, String> {
        let s = s.trim();

        // Remove type prefix if present
        let layers_str = if let Some(colon_pos) = s.find(':') {
            &s[colon_pos + 1..]
        } else {
            s
        };

        let mut layers = Vec::new();
        let mut current_num = String::new();

        for c in layers_str.chars() {
            if c.is_ascii_digit() {
                current_num.push(c);
            } else {
                if !current_num.is_empty() {
                    if let Ok(n) = current_num.parse::<u64>() {
                        layers.push(n);
                    }
                    current_num.clear();
                }
                // Also stop at '[' for activation notation like "256[relu]"
                if c == '[' {
                    break;
                }
            }
        }

        // Don't forget the last number
        if !current_num.is_empty() {
            if let Ok(n) = current_num.parse::<u64>() {
                layers.push(n);
            }
        }

        if layers.is_empty() {
            return Err("no layer sizes found in architecture".into());
        }

        Ok(layers)
    }
}

// =============================================================================
// UTILITY — constant expression evaluator (subset)
// =============================================================================

/// Attempt to evaluate a constant integer expression.
/// Returns `None` if the expression is dynamic or too complex.
fn eval_const_expr(expr: &Expr) -> Option<u64> {
    match expr {
        Expr::IntLit { value, .. } => Some(*value as u64),
        Expr::BinOp { op, lhs, rhs, .. } => {
            let l = eval_const_expr(lhs)?;
            let r = eval_const_expr(rhs)?;
            match op {
                BinOpKind::Add => Some(l + r),
                BinOpKind::Sub => l.checked_sub(r),
                BinOpKind::Mul => Some(l * r),
                BinOpKind::Div | BinOpKind::FloorDiv => {
                    if r != 0 {
                        Some(l / r)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

// =============================================================================
// UTILITY — numpy-style broadcast shape computation
// =============================================================================

/// Compute the broadcast result shape of two tensor shapes (numpy rules).
///
/// Shapes are compared right-to-left. Each pair of dimensions must be either:
///   • equal, or
///   • one of them is 1 (broadcastable), or
///   • one of them is Dynamic (result is Dynamic).
///
/// Returns `None` if the shapes are incompatible.
fn broadcast_shapes(a: &[Dim], b: &[Dim]) -> Option<Vec<Dim>> {
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);

    let pad_a = len - a.len();
    let pad_b = len - b.len();

    for i in 0..len {
        let da = if i < pad_a {
            &Dim::Lit(1)
        } else {
            &a[i - pad_a]
        };
        let db = if i < pad_b {
            &Dim::Lit(1)
        } else {
            &b[i - pad_b]
        };

        let out = match (da, db) {
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => Dim::Dynamic,
            (Dim::Named(_), _) | (_, Dim::Named(_)) => Dim::Dynamic,
            (Dim::Lit(x), Dim::Lit(y)) => {
                if x == y {
                    Dim::Lit(*x)
                } else if *x == 1 {
                    Dim::Lit(*y)
                } else if *y == 1 {
                    Dim::Lit(*x)
                } else {
                    return None; // incompatible
                }
            }
        };
        result.push(out);
    }
    Some(result)
}

// =============================================================================
// CONVENIENCE: top-level entry point
// =============================================================================

/// Check a Jules program and return all diagnostics.
///
/// ```rust,ignore
/// let diags = jules_check(&program);
/// for d in &diags.items {
///     eprintln!("[{}] {}: {}", format!("{:?}", d.severity), d.span, d.message);
/// }
/// ```
pub fn jules_check(program: &Program) -> Diagnostics {
    let mut ck = TypeCk::new();
    ck.check_program(program);
    ck.diag
}

fn is_network_decorator(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "ai" | "ppo" | "dqn" | "a2c" | "sac" | "ddpg" | "td3" | "trpo" | "reinforce"
    )
}

#[derive(Debug, Clone)]
struct AiDecoratorConfig {
    span: Span,
    network: String,
    learning_rate: Option<f64>,
    input: Option<u64>,
    output: Option<u64>,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::lexer::Span;

    fn dummy() -> Span {
        Span::dummy()
    }

    fn f32_ty() -> Type {
        Type::Scalar(ElemType::F32)
    }
    fn bool_ty() -> Type {
        Type::Scalar(ElemType::Bool)
    }

    fn tensor_ty(elem: ElemType, dims: Vec<u64>) -> Type {
        Type::Tensor {
            elem,
            shape: dims.into_iter().map(DimExpr::Lit).collect(),
        }
    }

    fn float_lit(v: f64) -> Expr {
        Expr::FloatLit {
            span: dummy(),
            value: v,
        }
    }
    fn int_lit(v: u128) -> Expr {
        Expr::IntLit {
            span: dummy(),
            value: v,
        }
    }
    fn bool_lit(v: bool) -> Expr {
        Expr::BoolLit {
            span: dummy(),
            value: v,
        }
    }
    fn ident(name: &str) -> Expr {
        Expr::Ident {
            span: dummy(),
            name: name.into(),
        }
    }

    fn make_checker() -> TypeCk {
        TypeCk::new()
    }

    fn mk_agent_with_attrs(attrs: Vec<Attribute>) -> AgentDecl {
        AgentDecl {
            span: dummy(),
            attrs,
            name: "Bot".into(),
            architecture: AgentArchitecture::Learned,
            perceptions: vec![PerceptionSpec {
                span: dummy(),
                kind: PerceptionKind::Vision,
                range: Some(128.0),
                fov: None,
                tag: Some("vision".into()),
            }],
            memory: None,
            learning: None,
            behaviors: vec![],
            goals: vec![],
            fields: vec![],
        }
    }

    // ── Scalar literal inference ───────────────────────────────────────────────

    #[test]
    fn test_int_lit_infers_i32() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        let ty = ck.check_expr(&int_lit(42), &mut env);
        assert_eq!(ty, Ty::Scalar(ElemType::I32));
        assert!(!ck.diag.has_errors());
    }

    #[test]
    fn test_float_lit_infers_f32() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        let ty = ck.check_expr(&float_lit(3.14), &mut env);
        assert_eq!(ty, Ty::Scalar(ElemType::F32));
        assert!(!ck.diag.has_errors());
    }

    // ── Undeclared variable ────────────────────────────────────────────────────

    #[test]
    fn test_undeclared_variable_error() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        ck.check_expr(&ident("x"), &mut env);
        assert!(ck.diag.has_errors());
    }

    // ── Let binding / type inference ──────────────────────────────────────────

    #[test]
    fn test_let_binding_infers_type() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        env.bind("x", Ty::Scalar(ElemType::F32));
        let ty = ck.check_expr(&ident("x"), &mut env);
        assert_eq!(ty, Ty::Scalar(ElemType::F32));
    }

    // ── Bool type checking ─────────────────────────────────────────────────────

    #[test]
    fn test_bool_lit() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        let ty = ck.check_expr(&bool_lit(true), &mut env);
        assert_eq!(ty, Ty::Bool);
    }

    // ── Matrix multiply shape inference ────────────────────────────────────────

    #[test]
    fn test_matmul_shape_correct() {
        // tensor<f32>[8, 16] @ tensor<f32>[16, 4] → tensor<f32>[8, 4]
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        env.bind(
            "A",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(8), Dim::Lit(16)],
            },
        );
        env.bind(
            "B",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(16), Dim::Lit(4)],
            },
        );
        let expr = Expr::MatMul {
            span: dummy(),
            lhs: Box::new(ident("A")),
            rhs: Box::new(ident("B")),
        };
        let ty = ck.check_expr(&expr, &mut env);
        assert!(!ck.diag.has_errors(), "{:?}", ck.diag.items);
        assert_eq!(
            ty,
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(8), Dim::Lit(4)],
            }
        );
    }

    #[test]
    fn test_matmul_shape_mismatch_error() {
        // tensor<f32>[8, 16] @ tensor<f32>[32, 4] — inner dim mismatch
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        env.bind(
            "A",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(8), Dim::Lit(16)],
            },
        );
        env.bind(
            "B",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(32), Dim::Lit(4)],
            },
        );
        let expr = Expr::MatMul {
            span: dummy(),
            lhs: Box::new(ident("A")),
            rhs: Box::new(ident("B")),
        };
        ck.check_expr(&expr, &mut env);
        assert!(
            ck.diag.has_errors(),
            "expected error for inner dim mismatch"
        );
    }

    // ── Element-wise (Hadamard) multiply ──────────────────────────────────────

    #[test]
    fn test_hadamard_same_shape_ok() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        let t = Ty::Tensor {
            elem: ElemType::F32,
            shape: vec![Dim::Lit(4)],
        };
        env.bind("A", t.clone());
        env.bind("B", t);
        let expr = Expr::HadamardMul {
            span: dummy(),
            lhs: Box::new(ident("A")),
            rhs: Box::new(ident("B")),
        };
        let ty = ck.check_expr(&expr, &mut env);
        assert!(!ck.diag.has_errors(), "{:?}", ck.diag.items);
        assert!(matches!(ty, Ty::Tensor { .. }));
    }

    // ── Tensor concat ─────────────────────────────────────────────────────────

    #[test]
    fn test_tensor_concat_axis0_shape() {
        // tensor[3, 8] ++ tensor[5, 8] → tensor[8, 8]
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        env.bind(
            "A",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(3), Dim::Lit(8)],
            },
        );
        env.bind(
            "B",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(5), Dim::Lit(8)],
            },
        );
        let expr = Expr::TensorConcat {
            span: dummy(),
            lhs: Box::new(ident("A")),
            rhs: Box::new(ident("B")),
        };
        let ty = ck.check_expr(&expr, &mut env);
        assert!(!ck.diag.has_errors(), "{:?}", ck.diag.items);
        assert_eq!(
            ty,
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(8), Dim::Lit(8)],
            }
        );
    }

    #[test]
    fn test_tensor_concat_inner_mismatch_error() {
        // tensor[3, 8] ++ tensor[5, 4] — dim 1 mismatch
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        env.bind(
            "A",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(3), Dim::Lit(8)],
            },
        );
        env.bind(
            "B",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(5), Dim::Lit(4)],
            },
        );
        let expr = Expr::TensorConcat {
            span: dummy(),
            lhs: Box::new(ident("A")),
            rhs: Box::new(ident("B")),
        };
        ck.check_expr(&expr, &mut env);
        assert!(ck.diag.has_errors(), "expected error for dim mismatch");
    }

    // ── Vec constructor ────────────────────────────────────────────────────────

    #[test]
    fn test_vec3_ctor_ok() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        let expr = Expr::VecCtor {
            span: dummy(),
            size: VecSize::N3,
            elems: vec![float_lit(1.0), float_lit(0.0), float_lit(0.0)],
        };
        let ty = ck.check_expr(&expr, &mut env);
        assert!(!ck.diag.has_errors(), "{:?}", ck.diag.items);
        assert_eq!(
            ty,
            Ty::Vec {
                size: VecSize::N3,
                family: VecFamily::Float
            }
        );
    }

    #[test]
    fn test_vec3_ctor_wrong_arity() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        let expr = Expr::VecCtor {
            span: dummy(),
            size: VecSize::N3,
            elems: vec![float_lit(1.0), float_lit(0.0)], // only 2
        };
        ck.check_expr(&expr, &mut env);
        assert!(ck.diag.has_errors(), "expected arity error");
    }

    // ── Field access on vec (swizzle) ─────────────────────────────────────────

    #[test]
    fn test_vec3_field_x_is_scalar() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        env.bind(
            "v",
            Ty::Vec {
                size: VecSize::N3,
                family: VecFamily::Float,
            },
        );
        let expr = Expr::Field {
            span: dummy(),
            object: Box::new(ident("v")),
            field: "x".into(),
        };
        let ty = ck.check_expr(&expr, &mut env);
        assert!(!ck.diag.has_errors(), "{:?}", ck.diag.items);
        assert_eq!(ty, Ty::Scalar(ElemType::F32));
    }

    // ── If expression type ─────────────────────────────────────────────────────

    #[test]
    fn test_if_expr_non_bool_condition_error() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        let expr = Expr::IfExpr {
            span: dummy(),
            cond: Box::new(int_lit(1)), // not bool
            then: Box::new(Block::new(dummy())),
            else_: None,
        };
        ck.check_expr(&expr, &mut env);
        assert!(ck.diag.has_errors());
    }

    // ── Grad requires float type ───────────────────────────────────────────────

    #[test]
    fn test_grad_on_non_float_error() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        env.bind("x", Ty::Scalar(ElemType::I32));
        let expr = Expr::Grad {
            span: dummy(),
            inner: Box::new(ident("x")),
        };
        ck.check_expr(&expr, &mut env);
        assert!(ck.diag.has_errors());
    }

    #[test]
    fn test_grad_on_float_tensor_ok() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        env.bind(
            "T",
            Ty::Tensor {
                elem: ElemType::F32,
                shape: vec![Dim::Lit(4), Dim::Lit(4)],
            },
        );
        let expr = Expr::Grad {
            span: dummy(),
            inner: Box::new(ident("T")),
        };
        let ty = ck.check_expr(&expr, &mut env);
        assert!(!ck.diag.has_errors(), "{:?}", ck.diag.items);
        assert!(matches!(ty, Ty::Tensor { .. }));
    }

    // ── Type environment scoping ───────────────────────────────────────────────

    #[test]
    fn test_scope_shadowing() {
        let mut env = TyEnv::new();
        env.bind("x", Ty::Scalar(ElemType::F32));
        env.push_scope();
        env.bind("x", Ty::Bool);
        assert_eq!(env.lookup("x"), Some(&Ty::Bool));
        env.pop_scope();
        assert_eq!(env.lookup("x"), Some(&Ty::Scalar(ElemType::F32)));
    }

    // ── Inference variable unification ─────────────────────────────────────────

    #[test]
    fn test_infer_ctx_unify_var_with_concrete() {
        let mut ctx = InferCtx::default();
        let v = ctx.fresh();
        let concrete = Ty::Scalar(ElemType::F64);
        assert!(ctx.unify(&v, &concrete));
        assert_eq!(ctx.resolve(&v), concrete);
    }

    #[test]
    fn test_infer_ctx_unify_two_concrete_compatible() {
        let mut ctx = InferCtx::default();
        let a = Ty::Scalar(ElemType::F32);
        let b = Ty::Scalar(ElemType::F32);
        assert!(ctx.unify(&a, &b));
    }

    #[test]
    fn test_infer_ctx_unify_incompatible() {
        let mut ctx = InferCtx::default();
        let a = Ty::Scalar(ElemType::F32);
        let b = Ty::Bool;
        assert!(!ctx.unify(&a, &b));
    }

    // ── Model dropout validation ───────────────────────────────────────────────

    #[test]
    fn test_model_dropout_out_of_range_error() {
        let m = ModelDecl {
            span: dummy(),
            attrs: vec![],
            name: "Bad".into(),
            layers: vec![ModelLayer::Dropout {
                span: dummy(),
                rate: 1.5,
            }],
            device: ModelDevice::Auto,
            optimizer: None,
        };
        let mut ck = make_checker();
        ck.check_model(&m);
        assert!(ck.diag.has_errors());
    }

    #[test]
    fn test_model_dropout_valid() {
        let m = ModelDecl {
            span: dummy(),
            attrs: vec![],
            name: "Good".into(),
            layers: vec![ModelLayer::Dropout {
                span: dummy(),
                rate: 0.1,
            }],
            device: ModelDevice::Auto,
            optimizer: None,
        };
        let mut ck = make_checker();
        ck.check_model(&m);
        assert!(!ck.diag.has_errors(), "{:?}", ck.diag.items);
    }

    #[test]
    fn test_model_attention_zero_heads_error() {
        let m = ModelDecl {
            span: dummy(),
            attrs: vec![],
            name: "Attn".into(),
            layers: vec![ModelLayer::Attention {
                span: dummy(),
                num_heads: 0,
                head_dim: 64,
            }],
            device: ModelDevice::Auto,
            optimizer: None,
        };
        let mut ck = make_checker();
        ck.check_model(&m);
        assert!(ck.diag.has_errors());
    }

    // ── System parameter missing type annotation ───────────────────────────────

    #[test]
    fn test_system_unannotated_param_error() {
        let s = SystemDecl::new(
            dummy(),
            "Update",
            vec![Param {
                span: dummy(),
                name: "dt".into(),
                ty: None,
                default: None,
                mutable: false,
            }],
            Block::new(dummy()),
        );
        let mut ck = make_checker();
        ck.check_system(&s);
        assert!(ck.diag.has_errors());
    }

    // ── Struct field lookup ────────────────────────────────────────────────────

    #[test]
    fn test_struct_field_access_ok() {
        let mut ck = make_checker();
        ck.symbols.structs.insert(
            "Pos".into(),
            StructInfo {
                name: "Pos".into(),
                fields: vec![("x".into(), Ty::Scalar(ElemType::F32))],
                is_component: false,
            },
        );
        let mut env = TyEnv::new();
        env.bind("p", Ty::Struct("Pos".into()));
        let expr = Expr::Field {
            span: dummy(),
            object: Box::new(ident("p")),
            field: "x".into(),
        };
        let ty = ck.check_expr(&expr, &mut env);
        assert_eq!(ty, Ty::Scalar(ElemType::F32));
        assert!(!ck.diag.has_errors(), "{:?}", ck.diag.items);
    }

    #[test]
    fn test_struct_field_access_missing_error() {
        let mut ck = make_checker();
        ck.symbols.structs.insert(
            "Pos".into(),
            StructInfo {
                name: "Pos".into(),
                fields: vec![("x".into(), Ty::Scalar(ElemType::F32))],
                is_component: false,
            },
        );
        let mut env = TyEnv::new();
        env.bind("p", Ty::Struct("Pos".into()));
        let expr = Expr::Field {
            span: dummy(),
            object: Box::new(ident("p")),
            field: "z".into(), // does not exist
        };
        ck.check_expr(&expr, &mut env);
        assert!(ck.diag.has_errors());
    }

    #[test]
    fn test_ai_decorator_named_options_valid() {
        let ai_attr = Attribute::Named {
            name: "ai".into(),
            args: vec![
                Expr::Assign {
                    span: dummy(),
                    op: AssignOpKind::Assign,
                    target: Box::new(Expr::Ident {
                        span: dummy(),
                        name: "network".into(),
                    }),
                    value: Box::new(Expr::StrLit {
                        span: dummy(),
                        value: "128->64->32".into(),
                    }),
                },
                Expr::Assign {
                    span: dummy(),
                    op: AssignOpKind::Assign,
                    target: Box::new(Expr::Ident {
                        span: dummy(),
                        name: "lr".into(),
                    }),
                    value: Box::new(Expr::FloatLit {
                        span: dummy(),
                        value: 0.0003,
                    }),
                },
                Expr::Assign {
                    span: dummy(),
                    op: AssignOpKind::Assign,
                    target: Box::new(Expr::Ident {
                        span: dummy(),
                        name: "input".into(),
                    }),
                    value: Box::new(Expr::IntLit {
                        span: dummy(),
                        value: 128,
                    }),
                },
                Expr::Assign {
                    span: dummy(),
                    op: AssignOpKind::Assign,
                    target: Box::new(Expr::Ident {
                        span: dummy(),
                        name: "output".into(),
                    }),
                    value: Box::new(Expr::IntLit {
                        span: dummy(),
                        value: 32,
                    }),
                },
            ],
        };

        let program = Program {
            span: dummy(),
            items: vec![Item::Agent(mk_agent_with_attrs(vec![ai_attr]))],
        };

        let mut ck = make_checker();
        ck.check_program(&program);
        assert!(
            !ck.diag.has_errors(),
            "unexpected errors for valid @ai options: {:?}",
            ck.diag.items
        );
    }

    #[test]
    fn test_ai_decorator_invalid_lr_error() {
        let ai_attr = Attribute::Named {
            name: "ai".into(),
            args: vec![
                Expr::StrLit {
                    span: dummy(),
                    value: "128->64->32".into(),
                },
                Expr::Assign {
                    span: dummy(),
                    op: AssignOpKind::Assign,
                    target: Box::new(Expr::Ident {
                        span: dummy(),
                        name: "lr".into(),
                    }),
                    value: Box::new(Expr::FloatLit {
                        span: dummy(),
                        value: -0.5,
                    }),
                },
            ],
        };

        let program = Program {
            span: dummy(),
            items: vec![Item::Agent(mk_agent_with_attrs(vec![ai_attr]))],
        };

        let mut ck = make_checker();
        ck.check_program(&program);
        assert!(ck.diag.has_errors());
    }

    // ── Diagnostic accumulation ────────────────────────────────────────────────

    #[test]
    fn test_multiple_errors_accumulated() {
        let mut ck = make_checker();
        let mut env = TyEnv::new();
        // Two undeclared variables — should produce two errors.
        ck.check_expr(&ident("a"), &mut env);
        ck.check_expr(&ident("b"), &mut env);
        assert_eq!(ck.diag.error_count(), 2);
    }

    // ── Diagnostic note attachment ─────────────────────────────────────────────

    #[test]
    fn test_diagnostic_note_attached() {
        let d = Diagnostic::error(dummy(), "oops").with_note(dummy(), "defined here");
        assert_eq!(d.notes.len(), 1);
        assert!(d.is_fatal());
    }

    // ── Ty::display formatting ─────────────────────────────────────────────────

    #[test]
    fn test_ty_display_tensor() {
        let ty = Ty::Tensor {
            elem: ElemType::F32,
            shape: vec![Dim::Lit(128), Dim::Lit(128)],
        };
        assert_eq!(ty.display(), "tensor<f32>[128, 128]");
    }

    #[test]
    fn test_ty_display_infer_var() {
        let ty = Ty::Infer(7);
        assert_eq!(ty.display(), "_#7");
    }
}
