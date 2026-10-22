"""Reference backends on an exported case: pgcore's CHOLMOD and UMFPACK, SciPy SuperLU.

usage: python ref_bench.py <case_dir> <out_dir> [--repeats R]
Appends one JSON line per backend to <out_dir>/ref_results.jsonl and writes
the solutions as <out_dir>/<case>__ref_<backend>.f64.

CHOLMOD is driven exactly as pgcore drives it in the DC forward
(LinSolver + setMatrix(S, 1)); one RHS per call, as pgcore solves.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spl

sys.path.insert(0, os.path.dirname(__file__))
import pygimli as pg

import caseio


def relres(A, X, B):
    out = []
    for j in range(B.shape[1]):
        bn = np.linalg.norm(B[:, j])
        r = np.linalg.norm(A @ X[:, j] - B[:, j])
        out.append(r / bn if bn > 0 else r)
    return float(np.max(out)) if np.all(np.isfinite(out)) else float("nan")


def main():
    case_dir, out_dir = sys.argv[1], sys.argv[2]
    R = int(sys.argv[sys.argv.index("--repeats") + 1]) if "--repeats" in sys.argv else 3
    os.makedirs(out_dir, exist_ok=True)
    A, B, Xref, meta = caseio.load_case(case_dir)
    name = os.path.basename(os.path.dirname(case_dir.rstrip("/\\"))) + "__" + os.path.basename(case_dir.rstrip("/\\"))
    n, m = B.shape
    S = pg.matrix.SparseMatrix(pg.core.IndexArray(A.indptr.astype(np.int64)),
                               pg.core.IndexArray(A.indices.astype(np.int64)),
                               pg.Vector(A.data))
    assert np.array_equal(np.array(S.vecVals()), A.data)
    threads = os.environ.get("OPENBLAS_NUM_THREADS", "default")
    rows = []

    def record(backend, t_fact, t_solve, X, extra=None):
        rec = dict(case=name, backend=backend, threads=threads, n=n, nnz=int(A.nnz), nrhs=m,
                   t_factor=t_fact, t_solve_cols=t_solve,
                   relres=relres(A, X, B) if X is not None else None,
                   finite=bool(np.isfinite(X).all()) if X is not None else None)
        if X is not None and Xref is not None and np.isfinite(X).all():
            d = np.linalg.norm(X - Xref, axis=0) / np.maximum(np.linalg.norm(Xref, axis=0), 1e-300)
            rec["reldiff_to_captured_cholmod"] = float(d.max())
        rec.update(extra or {})
        rows.append(rec)
        if X is not None:
            np.asfortranarray(X).T.ravel().astype("<f8").tofile(os.path.join(out_dir, f"{name}__ref_{backend}.f64"))
        print(json.dumps(rec), flush=True)

    def pg_backend(label, make):
        best = np.inf
        for _ in range(R):
            t0 = time.perf_counter()
            ls = make()
            best = min(best, time.perf_counter() - t0)
        X = np.empty((n, m))
        bs = [pg.Vector(B[:, j]) for j in range(m)]
        tb = np.inf
        for _ in range(R):
            t0 = time.perf_counter()
            for j in range(m):
                X[:, j] = np.array(ls.solve(bs[j]))
            tb = min(tb, time.perf_counter() - t0)
        record(label, best, tb, X, dict(solver_name=ls.solverName()))

    def chol():
        ls = pg.core.LinSolver(False)
        ls.setMatrix(S, 1)
        return ls

    backends = sys.argv[sys.argv.index("--backends") + 1].split(",") if "--backends" in sys.argv else ["cholmod", "umfpack", "superlu"]
    pg_backend("cholmod", chol)
    if "umfpack" in backends:
        pg_backend("umfpack", lambda: pg.core.LinSolver(S, pg.core.SolverType.UMFPACK, False))
    if "superlu" not in backends:
        with open(os.path.join(out_dir, "ref_results.jsonl"), "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return

    # SciPy SuperLU (BSD), COLAMD ordering
    Ac = A.tocsc()
    best = np.inf
    for _ in range(R):
        t0 = time.perf_counter()
        try:
            lu = spl.splu(Ac)
            err = None
        except RuntimeError as e:
            lu, err = None, str(e)
        best = min(best, time.perf_counter() - t0)
    if lu is None:
        record("superlu", best, None, None, dict(error=err))
    else:
        tb = np.inf
        for _ in range(R):
            t0 = time.perf_counter()
            X = lu.solve(B)
            tb = min(tb, time.perf_counter() - t0)
        # 1-norm condition estimate from the LU
        try:
            Ainv = spl.LinearOperator((n, n), matvec=lu.solve, rmatvec=lambda y: lu.solve(y, trans="T"),
                                      dtype=float)
            cond1 = float(spl.norm(Ac, 1) * spl.onenormest(Ainv))
        except Exception as e:  # noqa: BLE001
            cond1 = None
        record("superlu", best, None, X, dict(t_solve_block=tb, cond1_est=cond1,
                                               fill_LU=int(lu.L.nnz + lu.U.nnz)))
    with open(os.path.join(out_dir, "ref_results.jsonl"), "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
