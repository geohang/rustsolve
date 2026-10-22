"""Where the bench reads and writes. Override with environment variables.

RUSTSOLVE_BENCH_DIR  exported cases and results (large; default bench-data/ in the repo)
PHGX_REPO            PyHydroGeophysX checkout (its examples/data are the inputs)
CARGO_TARGET_DIR     where cargo put rustsolve_capi and faerbench
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
WORK = pathlib.Path(os.environ.get("RUSTSOLVE_BENCH_DIR", REPO / "bench-data"))
CASES = WORK / "cases"
RESULTS = WORK / "results"
PHGX = pathlib.Path(os.environ.get("PHGX_REPO", REPO.parent / "PyHydroGeophysX"))
DATA = PHGX / "examples" / "data"
TARGET = pathlib.Path(os.environ.get("CARGO_TARGET_DIR", REPO / "target"))
if os.name == "nt":
    CAPI = TARGET / "release" / "rustsolve_capi.dll"
    FAERBENCH = TARGET / "release" / "faerbench.exe"
else:
    CAPI = TARGET / "release" / ("librustsolve_capi.dylib" if os.uname().sysname == "Darwin" else "librustsolve_capi.so")
    FAERBENCH = TARGET / "release" / "faerbench"
