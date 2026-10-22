# bench

Accuracy and speed of rustsolve against reference solvers (CHOLMOD and
UMFPACK through an installed pyGIMLi, and SciPy's SuperLU) on systems
exported from finite-element models. The reference solvers are needed only
here; the crates do not depend on them. Results are in
[docs/benchmarks.md](../docs/benchmarks.md).

Paths come from `python/paths.py`:
- `RUSTSOLVE_BENCH_DIR`: exported cases and results (default `bench-data/`
  in the repository; the cases are large);
- `PHGX_REPO`: a PyHydroGeophysX checkout, whose example data the cases are
  made from (default: beside this repository);
- `CARGO_TARGET_DIR`: where cargo puts `faerbench`.

```bash
# 1. export the systems (each case in its own process, with a memory cap in GB)
python python/run_capped.py 3.5 python/export_cases.py line2 "$RUSTSOLVE_BENCH_DIR/cases"
#    cases: line2 e4d tl_synth contrast p2_primary grid3d failing scale3d_57k scale3d_97k
# 2. reference solvers
python python/ref_bench.py <case_dir> <out_dir> [--backends cholmod]
# 3. rustsolve
cargo run --release -p faerbench -- bench <case_dir> <out_dir> [--threads N] [--variants llt_auto,llt_nd,ldlt_auto,lu_auto]
# 4. tables
python python/compare.py <faer_dir> <ref_dir> <out.md> [<multithread_dir>]
```

A case directory holds `meta.json`, `indptr.i64`, `indices.i64`, `data.f64`
(the full symmetric matrix as CSR, little-endian), `rhs.f64` and the
reference solution `xref.f64` (n × nrhs, column-major).
