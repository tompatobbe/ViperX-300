#!/usr/bin/env python3
"""Python access to the paper authors' published M(q) and V(q,q̇) (MATLAB-only).

The paper repo (vendored at external/paper_model) ships its mass matrix and
Coriolis vector as huge symbolic MATLAB functions (Mass_Inertia_Matrix_function.m
≈ 360 kB, Coriolis_Vector_function.m ≈ 5.4 MB); only gravity was ported to
Python by the authors. Rather than hand-port ~80k lines (error-prone) or run
their MATLAB socket server, this module MECHANICALLY translates the .m source to
Python at load time. The files use a tiny MATLAB subset — scalar assignments,
element-wise operators, sin/cos, 1-based indexing/slicing of two parameter
vectors, and bracket literals — so the translation is a handful of rewrites:

    %…            → dropped (comments, incl. the commented-out alternatives)
    ...<newline>  → joined   (line continuation)
    .*  ./  .^  ^ → *  /  **  **
    X(k), X(a:b)  → X[k-1], X[a-1:b]   (only for the known array variables)
    [a; b] / [a, b; c, d] literals → np.array

Outputs keep the paper's native units: master-motor **mA** per rad/s² (M) and
mA (V). Joint angles are the robot joint frame (the q2/q3 DH offsets are applied
inside the paper's functions), exactly like `calculate_gravity` as used by
compare_gravity.py / tools/compare_paper_model.py.

NB V(q,q̇) is the Coriolis/centrifugal VECTOR (≙ C(q,q̇)q̇), not a matrix.

Usage:
    from tools.paper_dynamics import load_paper_M, load_paper_V
    M_fn = load_paper_M();  M_fn(q)      # (6,6) mA·s²/rad
    V_fn = load_paper_V();  V_fn(q, dq)  # (6,)  mA
"""
import os
import re

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(REPO, "external", "paper_model")
M_FILE = "Mass_Inertia_Matrix_function.m"
V_FILE = "Coriolis_Vector_function.m"
G_FILE = "Gravity_Compensationfunction.m"

_HEADER = re.compile(r"^\s*function\s+(\w+)\s*=\s*\w+\s*\(([^)]*)\)", re.M)
_LITERAL = re.compile(r"(\w+)\s*=\s*\[(.*?)\]\s*;", re.S)


def _literal_to_py(body):
    """MATLAB bracket literal body → numpy expression (rows split on ';'/newline)."""
    rows = [r for r in re.split(r"[;\n]", body) if r.strip()]
    cells = [[c for c in re.split(r"[,\s]+", r.strip()) if c] for r in rows]
    if all(len(r) == 1 for r in cells):                     # column vector
        return "np.array([" + ", ".join(r[0] for r in cells) + "])"
    return "np.array([" + ", ".join("[" + ", ".join(r) + "]" for r in cells) + "])"


_EXP_TAIL = re.compile(r"(?<![A-Za-z_\d.])\d+\.?\d*[eE]$")   # "1.5e" of 1.5e+01


def _chunk_sum(stmt, chunk=200):
    """Split `X = t1 + t2 − …` into accumulating statements.

    The Coriolis expressions are sums of tens of thousands of terms; CPython
    parses a+b+c+… as a left-deep tree and overflows its recursion limit while
    compiling. Cutting at depth-0 binary +/− (sign kept with its term) and
    accumulating `X = X + (…)` in fixed-size chunks is algebraically identical.
    """
    lhs, sep, rhs = stmt.partition("=")
    if not sep or len(rhs) < 20000:
        return [stmt]
    terms, depth, start = [], 0, 0
    for i, ch in enumerate(rhs):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch in "+-" and depth == 0 and i > start:
            prev = rhs[start:i].rstrip()
            if prev and (prev[-1].isalnum() or prev[-1] in ")]") \
                    and not _EXP_TAIL.search(prev):
                terms.append(rhs[start:i]); start = i
    terms.append(rhs[start:])
    lhs = lhs.strip()
    out = [f"{lhs} = 0.0"]
    for k in range(0, len(terms), chunk):
        out.append(f"{lhs} = {lhs} + ({''.join(terms[k:k + chunk])})")
    return out


def translate(src):
    """Translate one paper .m function to Python source. Returns (py_src, out, args)."""
    src = re.sub(r"%[^\n]*", "", src)              # comments (no strings in these files)
    src = re.sub(r"\.\.\.[ \t]*\n", " ", src)       # line continuations
    m = _HEADER.search(src)
    out_name, args = m.group(1), [a.strip() for a in m.group(2).split(",")]
    body = src[m.end():]
    body = re.sub(r"\bend\s*$", "", body.strip())   # trailing `end`

    # Bracket literals → np.array; remember which names are arrays (1-based index).
    arrays = set()

    def lit(mm):
        arrays.add(mm.group(1))
        return f"{mm.group(1)} = {_literal_to_py(mm.group(2))}\n"
    body = _LITERAL.sub(lit, body)
    # Slices of arrays produce arrays too (Parameters_k = coefficients(a:b)).
    for name, base in re.findall(r"(\w+)\s*=\s*(\w+)\(\d+:\d+\)", body):
        arrays.add(name)
    # A scalar element that is itself indexed later (`P1 = c(1); x = P1(1)`) is a
    # MATLAB 1×1 array: take it as a length-1 slice so the later [0] works.
    for name, base, k in re.findall(r"(\w+)\s*=\s*(\w+)\((\d+)\)\s*;", body):
        if base in arrays and re.search(rf"\b{name}\(", body):
            body = body.replace(f"{name} = {base}({k})", f"{name} = {base}({k}:{k})")
            arrays.add(name)

    alt = "|".join(sorted(arrays, key=len, reverse=True))
    body = re.sub(rf"\b({alt})\((\d+):(\d+)\)",
                  lambda k: f"{k.group(1)}[{int(k.group(2)) - 1}:{k.group(3)}]", body)
    body = re.sub(rf"\b({alt})\((\d+)\)",
                  lambda k: f"{k.group(1)}[{int(k.group(2)) - 1}]", body)
    body = body.replace(".*", "*").replace("./", "/").replace(".^", "**")
    body = body.replace("^", "**").replace("deg2rad", "np.deg2rad")

    lines = []
    for stmt in re.split(r"[;\n]", body):
        stmt = stmt.strip()
        if stmt:
            lines.extend("    " + s for s in _chunk_sum(stmt))
    py = f"def _f({', '.join(args)}):\n" + "\n".join(lines) + f"\n    return {out_name}\n"
    return py, out_name, args


def _load(fname, paper_dir=PAPER_DIR):
    with open(os.path.join(paper_dir, fname)) as fh:
        py, _, _ = translate(fh.read())
    ns = {"np": np, "sin": np.sin, "cos": np.cos}
    exec(compile(py, f"<translated {fname}>", "exec"), ns)
    return ns["_f"]


def load_paper_M(paper_dir=PAPER_DIR):
    """Paper mass matrix M(q) → (6,6) array [mA·s²/rad]."""
    f = _load(M_FILE, paper_dir)
    return lambda q: np.asarray(f(*np.asarray(q, float)), float).reshape(6, 6)


def load_paper_V(paper_dir=PAPER_DIR):
    """Paper Coriolis/centrifugal vector V(q, q̇) ≙ C(q,q̇)q̇ → (6,) array [mA]."""
    f = _load(V_FILE, paper_dir)
    return lambda q, dq: np.asarray(
        f(*np.asarray(q, float), *np.asarray(dq, float)), float).reshape(6)


def load_paper_G(paper_dir=PAPER_DIR):
    """Paper gravity G(q) translated from the .m [mA]. The authors also ship a
    hand port (Gravity_Compensation_Function.py); agreement of the two is the
    end-to-end check of this translator."""
    f = _load(G_FILE, paper_dir)
    return lambda q: np.asarray(f(*np.asarray(q, float)), float).reshape(6)


if __name__ == "__main__":
    import time
    t0 = time.time(); M_fn = load_paper_M(); t1 = time.time()
    V_fn = load_paper_V(); t2 = time.time()
    q = np.array([0.0, -0.6, 0.5, 0.0, 0.4, 0.0]); dq = np.full(6, 0.5)
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    print(f"translated M in {t1 - t0:.1f} s, V in {t2 - t1:.1f} s")
    print("M(q) [mA·s²/rad]:\n", M_fn(q))
    print("V(q,q̇) [mA]:", V_fn(q, dq))
