#!/usr/bin/env python3
"""
Generate many unique Jules functions for testing, fuzzing, or demos.

This enhanced generator can optionally "learn" repository syntax by scanning
.rs files and extracting tokens (Value variants, public functions/structs,
component string literals, and keywords) to produce more topical Jules code.

Usage examples:
  python3 scripts/generate_jules.py -n 1000 -o generated.jules --seed 42 --call-main 50
  python3 scripts/generate_jules.py --learn-syntax --learn-root . -n 2000 -o gen.jules
  python3 scripts/generate_jules.py --learn-syntax --dump-syntax
"""
import argparse
import random
import os
import sys
import re
from textwrap import indent
from typing import Dict, Any


INT_LITS = list(range(0, 256))
FLOAT_LITS = [round(i * 0.125, 3) for i in range(0, 256)]


def gen_literal(is_float):
    if is_float:
        return f"{random.choice(FLOAT_LITS):.3f}"
    else:
        return str(random.choice(INT_LITS))


def gen_var_name(fn_index, vindex):
    return f"v_{fn_index}_{vindex}"


def gen_assignment(var, is_float):
    op = random.choice(["+", "-", "*"])
    lit = gen_literal(is_float)
    return f"{var} = {var} {op} {lit};"


def gen_if_block(var, is_float):
    lit = gen_literal(is_float)
    lit2 = gen_literal(is_float)
    body = []
    body.append(f"if {var} > {lit} {{")
    body.append(f"    {var} = {var} - {lit2};")
    body.append("} else {")
    body.append(f"    {var} = {var} + {lit2};")
    body.append("}")
    return "\n".join(body)


def gen_for_loop(var, is_float):
    # iterate a small range and update var each iteration
    r = random.randint(2, 20)
    lit = gen_literal(is_float)
    body = []
    body.append(f"for i in 0..{r} {{")
    if is_float:
        body.append(f"    {var} = {var} + {lit};")
    else:
        body.append(f"    {var} = {var} + i + {lit};")
    body.append("}")
    return "\n".join(body)


def scan_repo_for_syntax(root: str) -> Dict[str, Any]:
    """Scan .rs files under `root` and extract tokens useful for generation.

    Returns a dict with keys: variants, pub_fns, pub_structs, components, strings, keywords
    """
    agg_text = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip target directory to save time
        if 'target' in dirpath.split(os.sep):
            continue
        for fn in filenames:
            if fn.endswith('.rs'):
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                        agg_text.append(fh.read())
                except Exception:
                    continue
    concat = "\n".join(agg_text)
    out = {}
    # Value enum variants
    m = re.search(r'pub\s+enum\s+Value\s*\{([^}]*)\}', concat, re.S)
    variants = []
    if m:
        block = m.group(1)
        # capture lines that start with an identifier
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            # stop at comment or doc
            if line.startswith('//'):
                continue
            vm = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', line)
            if vm:
                variants.append(vm.group(1))
    out['variants'] = list(dict.fromkeys(variants))

    out['pub_fns'] = list(set(re.findall(r'pub\s+fn\s+([a-zA-Z0-9_]+)\s*\(', concat)))
    out['pub_structs'] = list(set(re.findall(r'pub\s+struct\s+([A-Za-z0-9_]+)', concat)))

    comps = re.findall(r'insert_component\([^,]*,\s*"([^"]+)"', concat)
    out['components'] = list(dict.fromkeys(comps))

    strings = re.findall(r'"([^"\n]{1,80})"', concat)
    out['strings'] = list(dict.fromkeys(strings))

    # detect high-level keywords
    kwlist = ['tensor', 'matmul', 'simd', 'gpu', 'shader', 'spawn', 'insert_component', 'bench', 'scene', 'profil', 'network', 'udp', 'hot_reload', 'entity']
    found = []
    for kw in kwlist:
        if re.search(r'\b' + re.escape(kw) + r'\b', concat, re.I):
            found.append(kw)
    out['keywords'] = found
    return out


def gen_function(fn_index, min_stmts=3, max_stmts=10, learn: Dict[str, Any]=None):
    name = f"f_{fn_index:06d}"
    stmts = []
    # choose number of local variables
    num_vars = random.randint(1, 5)
    vars_info = []
    for vi in range(num_vars):
        is_mut = random.random() < 0.7
        is_float = random.random() < 0.45
        vname = gen_var_name(fn_index, vi)
        init = gen_literal(is_float)
        if is_mut:
            stmts.append(f"let mut {vname} = {init};")
        else:
            stmts.append(f"let {vname} = {init};")
        vars_info.append((vname, is_float, is_mut))

    # optionally bias toward topics discovered in repository
    topic = 'generic'
    if learn:
        if learn.get('components') and random.random() < 0.5:
            topic = 'ecs'
        elif 'tensor' in learn.get('keywords', []) and random.random() < 0.4:
            topic = 'nn'
        elif 'shader' in learn.get('keywords', []) and random.random() < 0.3:
            topic = 'shader'
        elif 'network' in learn.get('keywords', []) and random.random() < 0.25:
            topic = 'net'

    nstm = random.randint(min_stmts, max_stmts)
    for _ in range(nstm):
        if topic == 'ecs':
            # generate component-oriented statements using discovered component names
            comps = learn.get('components', []) if learn else []
            if comps:
                comp = random.choice(comps)
                # choose a mutable or immutable var to update
                v = random.choice(vars_info)
                if v[2]:
                    stmts.append(f"// ECS: attach/modify component {comp}")
                    stmts.append(f"insert_component(entity_id, \"{comp}\", {v[0]});")
                else:
                    stmts.append(f"let tmp = {v[0]}; // read-only component sample")
            else:
                # fallback to generic
                kind = random.choice(['assign', 'for', 'println'])
                if kind == 'assign':
                    v = random.choice(vars_info)
                    if v[2]:
                        stmts.append(gen_assignment(v[0], v[1]))
                    else:
                        stmts.append(f"let tmp_{v[0]} = {v[0]} + {gen_literal(v[1])};")
                elif kind == 'for':
                    v = random.choice(vars_info)
                    if v[2]:
                        stmts.append(gen_for_loop(v[0], v[1]))
                    else:
                        tmp = f"t_{v[0]}"
                        stmts.append(f"let mut {tmp} = {v[0]};")
                        stmts.append(gen_for_loop(tmp, v[1]))
                else:
                    v = random.choice(vars_info)
                    stmts.append(f"println(\"{name}: {v[0]}\");")
        elif topic == 'nn':
            # neural-network like snippets
            v = random.choice(vars_info)
            stmts.append(f"// NN micro-op: matrix multiply placeholder")
            stmts.append(f"let a = zeros(4,4);")
            stmts.append(f"let b = zeros(4,4);")
            stmts.append(f"let c = matmul(a, b);")
            stmts.append(f"println(\"{name} matmul done\");")
        elif topic == 'shader':
            # embed small shader snippet
            shader_name = f"shader_{fn_index:06d}"
            stmts.append(f"let src = \"void main() {{}}\";")
            stmts.append(f"let _ = compile_shader(\"{shader_name}\", src);")
        elif topic == 'net':
            v = random.choice(vars_info)
            stmts.append(f"// network send (stub)")
            stmts.append(f"send_one(\"127.0.0.1:12345\", \"ping\");")
        else:
            # generic behavior
            kind = random.choices(['assign', 'if', 'for', 'println'], [0.5, 0.15, 0.25, 0.1])[0]
            if kind == 'assign':
                v = random.choice(vars_info)
                if not v[2]:
                    lit = gen_literal(v[1])
                    stmts.append(f"let tmp_{v[0]} = {v[0]} + {lit};")
                else:
                    stmts.append(gen_assignment(v[0], v[1]))
            elif kind == 'if':
                v = random.choice(vars_info)
                if v[2]:
                    stmts.append(gen_if_block(v[0], v[1]))
                else:
                    tmp = f"t_{v[0]}"
                    stmts.append(f"let mut {tmp} = {v[0]};")
                    stmts.append(gen_if_block(tmp, v[1]))
            elif kind == 'for':
                v = random.choice(vars_info)
                if v[2]:
                    stmts.append(gen_for_loop(v[0], v[1]))
                else:
                    tmp = f"t_{v[0]}"
                    stmts.append(f"let mut {tmp} = {v[0]};")
                    stmts.append(gen_for_loop(tmp, v[1]))
            else:
                v = random.choice(vars_info)
                if random.random() < 0.5:
                    stmts.append(f"println(\"{name}: {v[0]}\", {v[0]});")
                else:
                    stmts.append(f"println(\"{name}\");")

    # final println to make observable side-effect
    last_var = random.choice(vars_info)[0]
    # final observable print using two-argument form
    stmts.append(f'println("{name} result:", {last_var});')

    body = "\n".join(indent(s, "    ") for s in stmts)
    func = f"fn {name}() {{\n{body}\n}}\n\n"
    return func


def generate_file(path, count, seed=None, call_main=0, min_stmts=3, max_stmts=10, learn_root=None, dump_syntax=False):
    learn_data = None
    if learn_root:
        learn_data = scan_repo_for_syntax(learn_root)
        if dump_syntax:
            print("Learned syntax:")
            for k, v in learn_data.items():
                print(f"{k}: {v[:10] if isinstance(v, list) else v}")
    if seed is not None:
        random.seed(seed)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"// Generated Jules: {count} functions (seed={seed})\n\n")
        for i in range(1, count + 1):
            f.write(gen_function(i, min_stmts=min_stmts, max_stmts=max_stmts, learn=learn_data))
        if call_main and call_main > 0:
            calls = []
            for i in range(1, min(call_main, count) + 1):
                calls.append(f"    f_{i:06d}();")
            main_body = "\n".join(calls)
            f.write(f"fn main() {{\n{main_body}\n}}\n")


def main():
    ap = argparse.ArgumentParser(description="Generate Jules functions")
    ap.add_argument("-n", "--num", type=int, default=1000, help="number of functions to generate")
    ap.add_argument("-o", "--out", default="generated.jules", help="output file path")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--call-main", type=int, default=0, help="emit a main that calls first N functions")
    ap.add_argument("--min-stmts", type=int, default=3)
    ap.add_argument("--max-stmts", type=int, default=8)
    ap.add_argument("--learn-syntax", action='store_true', help="scan repo .rs files and learn syntax tokens to include in generation")
    ap.add_argument("--learn-root", default='.', help="root path to scan for .rs files when --learn-syntax is used")
    ap.add_argument("--dump-syntax", action='store_true', help="print the learned syntax and exit")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    learn_root = args.learn_root if args.learn_syntax else None
    generate_file(args.out, args.num, seed=args.seed, call_main=args.call_main, min_stmts=args.min_stmts, max_stmts=args.max_stmts, learn_root=learn_root, dump_syntax=args.dump_syntax)
    print(f"Wrote {args.num} functions to {args.out}")


if __name__ == "__main__":
    main()
