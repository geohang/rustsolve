"""Summarise faer and reference runs into markdown tables.

usage: python compare.py <faer_dir> <ref_dir> <out.md> [<mt_dir>]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import caseio

import paths

ROOT = str(paths.WORK)
FAMILY_NOTE = {
    "line2": "quick-start inversion, fielddataline2 (2D, topography, SR + H2), 3 responses x 18 k",
    "e4d": "E4D 2021-10-08 inversion (2D, 112 electrodes, topography)",
    "tl_synth": "synthetic one-year time lapse, days 30/150/210/330 (2D, 24 k nodes)",
    "contrast": "line2 mesh, host 1e4 ohm-m: body 1, body 1e-3, checkerboard 1/1e4",
    "p2_primary": "P2 forward mesh (topography primary potential), homogeneous",
    "grid3d": "3D tet grid, DC forward (pgcore, calibration node)",
    "scale3d": "3D jittered tet grid, P1 stiffness + Dirichlet (scaling only)",
    "failing": "failure cases",
}


def load_jsonl(p):
    out = []
    if os.path.exists(p):
        with open(p) as f:
            out = [json.loads(line) for line in f if line.strip()]
    return out


def sol(d, case, tag, n, m):
    p = os.path.join(d, f"{case}__{tag}.f64")
    return np.fromfile(p, dtype="<f8").reshape(m, n).T if os.path.exists(p) else None


def reldiff(X, Y):
    if X is None or Y is None or not np.isfinite(X).all() or not np.isfinite(Y).all():
        return np.nan
    return float((np.linalg.norm(X - Y, axis=0) / np.maximum(np.linalg.norm(Y, axis=0), 1e-300)).max())


def g(x):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.1e}"


def ms(x):
    return "–" if x is None or not np.isfinite(x) else (f"{1e3 * x:.1f}" if x >= 0.01 else f"{1e3 * x:.2f}")


def nanmax(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return max(v) if v else np.nan


def nanmin(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return min(v) if v else np.nan


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else np.nan


VARIANTS = ("llt_auto", "llt_nd", "llt_simplicial", "llt_supernodal", "ldlt_auto", "lu_auto")


def main():
    fdir, rdir, out_md = sys.argv[1], sys.argv[2], sys.argv[3]
    F = {(r["case"], r["variant"]): r for r in load_jsonl(os.path.join(fdir, "results.jsonl"))}
    # simplicial/supernodal only ran in the first pass
    for r in load_jsonl(os.path.join(rdir, "results.jsonl")):
        F.setdefault((r["case"], r["variant"]), dict(r, _dir=rdir))
    Rf = {(r["case"], r["backend"]): r for r in load_jsonl(os.path.join(rdir, "ref_results.jsonl"))}
    cases = sorted({c for c, _ in F} | {c for c, _ in Rf})
    rows = []
    for c in cases:
        fam, name = c.split("__", 1)
        A, B, Xref, meta = caseio.load_case(os.path.join(ROOT, "cases", fam, name))
        n, m = B.shape
        row = dict(case=c, family=fam, n=n, nnz=int(A.nnz), nrhs=m, cond1=Rf.get((c, "superlu"), {}).get("cond1_est"))
        for b in ("cholmod", "umfpack", "superlu"):
            r = Rf.get((c, b), {})
            for k in ("relres", "t_factor", "t_solve_cols", "reldiff_to_captured_cholmod", "finite", "t_solve_block"):
                row[f"{b}_{k}"] = r.get(k)
        for v in VARIANTS:
            r = F.get((c, v), {})
            row[f"{v}_err"] = (r.get("error") or {}).get("code")
            row[f"{v}_errdetail"] = (r.get("error") or {}).get("detail")
            for k in ("error_stage", "relres", "t_analyze", "t_factor", "t_refactor", "t_solve_block", "t_solve_cols",
                      "supernodal", "min_pivot", "max_pivot", "pivot_ratio", "neg_pivots", "rcond_est", "len_l",
                      "solve_after_error"):
                row[f"{v}_{k}"] = r.get(k)
            X = sol(r.get("_dir", fdir), c, v, n, m) if r and not row[f"{v}_err"] else None
            row[f"{v}_diff"] = reldiff(X, Xref) if Xref is not None else np.nan
        rows.append(row)
    with open(out_md.replace(".md", "_rows.json"), "w") as f:
        json.dump(rows, f, indent=0, default=float)

    fams = defaultdict(list)
    for r in rows:
        fams[r["family"]].append(r)
    o = []
    w = o.append
    w("### Families\n")
    w("| family | what | systems | n | nnz(A) | RHS per factorization |")
    w("|---|---|---|---|---|---|")
    for fam, rs in fams.items():
        w(f"| {fam} | {FAMILY_NOTE.get(fam, '')} | {len(rs)} | {min(r['n'] for r in rs)}–{max(r['n'] for r in rs)} | "
          f"{min(r['nnz'] for r in rs)}–{max(r['nnz'] for r in rs)} | {max(r['nrhs'] for r in rs)} |")

    w("\n### Accuracy (worst system of each family, worst RHS column)\n")
    w("relres = ‖Ax−b‖/‖b‖. diff = ‖x−x_CHOLMOD‖/‖x_CHOLMOD‖, against the solution CHOLMOD returned inside pgcore's forward. "
      "cond = 1-norm estimate (SuperLU + onenormest); rcond = faer's Hager–Higham estimate (1/cond).\n")
    w("| family | cond (SuperLU est.) | 1/rcond (faer est.) | relres CHOLMOD | relres faer LLT (AMD) | relres faer LLT (ND) | relres faer LU | diff faer LLT | diff faer LLT ND | diff UMFPACK | diff SuperLU |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for fam, rs in fams.items():
        if fam == "failing":
            continue
        w(f"| {fam} | {g(nanmin([r['cond1'] for r in rs]))}–{g(nanmax([r['cond1'] for r in rs]))} | "
          f"{g(nanmax([1 / r['llt_auto_rcond_est'] if r['llt_auto_rcond_est'] else np.nan for r in rs]))} | "
          f"{g(nanmax([r['cholmod_relres'] for r in rs]))} | {g(nanmax([r['llt_auto_relres'] for r in rs]))} | "
          f"{g(nanmax([r['llt_nd_relres'] for r in rs]))} | {g(nanmax([r['lu_auto_relres'] for r in rs]))} | "
          f"{g(nanmax([r['llt_auto_diff'] for r in rs]))} | {g(nanmax([r['llt_nd_diff'] for r in rs]))} | "
          f"{g(nanmax([r['umfpack_reldiff_to_captured_cholmod'] for r in rs]))} | {g(nanmax([r['superlu_reldiff_to_captured_cholmod'] for r in rs]))} |")

    w("\n### Speed, one thread (median over the family, ms)\n")
    w("CHOLMOD = pgcore's LinSolver (analysis and factorization in one call, as pgcore makes it for every matrix). "
      "faer refactor = numeric factorization on a kept analysis. Solves: pgcore's one RHS per call (CHOLMOD through Python, so ~10 µs/call overhead), faer one per call and all RHS as one block.\n")
    w("| family | n | nnz(L) AMD | nnz(L) ND | CHOLMOD analyze+factor | faer analyze (AMD) | faer factor | faer refactor | faer analyze (ND) | faer factor (ND) | UMFPACK factor | SuperLU factor | CHOLMOD solves | faer solves, one per call | faer solves, block |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fam, rs in fams.items():
        if fam == "failing":
            continue
        M = lambda k, rr=rs: med([r[k] for r in rr])
        w(f"| {fam} | {int(M('n'))} | {int(M('llt_auto_len_l'))} | {int(M('llt_nd_len_l')) if np.isfinite(M('llt_nd_len_l')) else '–'} | "
          f"{ms(M('cholmod_t_factor'))} | {ms(M('llt_auto_t_analyze'))} | {ms(M('llt_auto_t_factor'))} | {ms(M('llt_auto_t_refactor'))} | "
          f"{ms(M('llt_nd_t_analyze'))} | {ms(M('llt_nd_t_factor'))} | {ms(M('umfpack_t_factor'))} | {ms(M('superlu_t_factor'))} | "
          f"{ms(M('cholmod_t_solve_cols'))} | {ms(M('llt_auto_t_solve_cols'))} | {ms(M('llt_auto_t_solve_block'))} |")

    w("\n### faer variants, one thread (median ms: analyze / factor / block solve)\n")
    w("| family | LLT auto (AMD) | LLT ND | LLT simplicial | LLT supernodal | LDLT | LU |")
    w("|---|---|---|---|---|---|---|")
    for fam, rs in fams.items():
        if fam == "failing":
            continue
        cells = []
        for v in VARIANTS:
            ok = [r for r in rs if r[f"{v}_t_factor"] is not None and not r[f"{v}_err"]]
            cells.append("–" if not ok else " / ".join(ms(med([r[f"{v}_{k}"] for r in ok])) for k in ("t_analyze", "t_factor", "t_solve_block")))
        w(f"| {fam} | " + " | ".join(cells) + " |")

    k = [r for r in rows if r["case"].startswith("line2__resp01_k")]
    if k:
        chol = sum(r["cholmod_t_factor"] for r in k)
        fa_naive = sum(r["llt_auto_t_analyze"] + r["llt_auto_t_factor"] for r in k)
        fa = k[0]["llt_auto_t_analyze"] + k[0]["llt_auto_t_factor"] + sum(r["llt_auto_t_refactor"] for r in k[1:])
        w(f"\n### One 2.5D response of line2 ({len(k)} wavenumbers x {k[0]['nrhs']} RHS), one thread\n")
        w(f"- factorizations: CHOLMOD as pgcore calls it {1e3 * chol:.1f} ms; faer called the same way {1e3 * fa_naive:.1f} ms; "
          f"faer with one analysis and {len(k) - 1} refactorizations {1e3 * fa:.1f} ms")
        w(f"- solves: CHOLMOD one RHS per call {1e3 * sum(r['cholmod_t_solve_cols'] for r in k):.1f} ms; faer one per call "
          f"{1e3 * sum(r['llt_auto_t_solve_cols'] for r in k):.1f} ms; faer as one block {1e3 * sum(r['llt_auto_t_solve_block'] for r in k):.1f} ms")

    w("\n### Pivot ratio min|d|/max|d| of faer LLT (the singularity check refuses below n·eps)\n")
    w("| family | smallest ratio | n·eps |")
    w("|---|---|---|")
    for fam, rs in fams.items():
        v = [(r["llt_auto_pivot_ratio"], r["n"]) for r in rs if r["llt_auto_pivot_ratio"] is not None]
        if v:
            pr, n = min(v)
            w(f"| {fam} | {g(pr)} | {g(n * np.finfo(float).eps)} |")

    w("\n### Failure behaviour\n")
    w("| system | CHOLMOD (pgcore LinSolver) | UMFPACK | SuperLU | faer LLT | faer LDLT | faer LU |")
    w("|---|---|---|---|---|---|---|")
    for r in fams.get("failing", []):
        def ref_cell(b):
            if r[f"{b}_t_factor"] is None:
                return "–"
            if r[f"{b}_finite"] is False:
                return "no error; NaN/Inf solution"
            return f"no error; relres {g(r[f'{b}_relres'])}"

        def faer_cell(v):
            if r[f"{v}_err"]:
                after = r[f"{v}_solve_after_error"]
                return f"**{r[f'{v}_err']}**" + (f"; later solve: {after}" if after else "")
            if r[f"{v}_relres"] is None:
                return "–"
            extra = f", neg. pivots {r[f'{v}_neg_pivots']}" if v == "ldlt_auto" else ""
            pr = "" if v == "lu_auto" else f", pivot ratio {g(r[f'{v}_pivot_ratio'])}"
            return f"OK; relres {g(r[f'{v}_relres'])}{pr}{extra}"
        w(f"| {r['case'].split('__')[1]} | {ref_cell('cholmod')} | {ref_cell('umfpack')} | {ref_cell('superlu')} | "
          f"{faer_cell('llt_auto')} | {faer_cell('ldlt_auto')} | {faer_cell('lu_auto')} |")

    if len(sys.argv) > 4:
        mt = sys.argv[4]
        Fm = load_jsonl(os.path.join(mt, "results.jsonl"))
        Rm = {r["case"]: r for r in load_jsonl(os.path.join(mt, "ref_results.jsonl"))}
        w("\n### Multithreaded (ms: analyze+factor; CHOLMOD with OpenBLAS's default threads)\n")
        w("| system | n | CHOLMOD | faer AMD 4 thr | faer AMD 10 | faer AMD 20 | faer ND 4 | faer ND 10 | faer ND 20 | faer ND 20 refactor |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        by = defaultdict(dict)
        for r in Fm:
            by[r["case"]][(r["variant"], r["threads"])] = r
        for c, d in by.items():
            if not (c.startswith("scale3d") or c.startswith("grid3d") or c.endswith("day210_k00") or c.endswith("day030_k00")):
                continue
            cell = lambda v, t: ms(d[(v, t)]["t_analyze"] + d[(v, t)]["t_factor"]) if (v, t) in d and "t_factor" in d[(v, t)] else "–"
            refac = ms(d[("llt_nd", 20)]["t_refactor"]) if ("llt_nd", 20) in d else "–"
            w(f"| {c} | {Rm.get(c, {}).get('n', '')} | {ms(Rm.get(c, {}).get('t_factor', np.nan))} | {cell('llt_auto', 4)} | {cell('llt_auto', 10)} | "
              f"{cell('llt_auto', 20)} | {cell('llt_nd', 4)} | {cell('llt_nd', 10)} | {cell('llt_nd', 20)} | {refac} |")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(o) + "\n")
    print("wrote", out_md)


if __name__ == "__main__":
    main()
