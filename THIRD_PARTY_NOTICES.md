# Third-party notices

rustsolve is MIT OR Apache-2.0. A build includes the Rust crates below, under
their own licences (all permissive). None of them contains SuiteSparse's
CHOLMOD or UMFPACK, or other GPL code. `cargo tree -e normal` lists the exact
set for a given build.

| crate | licence | notes |
|---|---|---|
| faer 0.24.4, faer-traits | MIT | faer also ships the licences of code it ports: AMD and COLAMD from SuiteSparse (BSD-3-Clause; Amestoy, Davis, Duff), parts of LAPACK (BSD), and Eigen's divide-and-conquer SVD (MPL-2.0), which rustsolve does not call |
| gemm, gemm-*, nano-gemm*, private-gemm-x86, pulp, dyn-stack, reborrow, equator, qd, spindle, raw-cpuid, libm | MIT | faer's kernels |
| rayon, rayon-core, crossbeam-*, either, num-complex, num-traits, num_cpus, once_cell, bytemuck, generativity, seq-macro, paste, windows-sys | MIT OR Apache-2.0 (bytemuck also Zlib) | |
| atomic-wait | BSD-2-Clause | |
| feral-metis, feral-ordering-core (feature `nd`) | MIT | multilevel nested dissection after Karypis and Kumar (1998) |
| feral-amd (feature `nd`) | MIT | approximate minimum degree after Amestoy, Davis and Duff; by its README a Rust transliteration of SuiteSparse's AMD (BSD-3-Clause), checked against faer's port |
| pyo3, numpy (Python package only) | MIT OR Apache-2.0 / BSD-2-Clause | |

Build-time only (procedural macros, not in the binaries): syn, quote,
proc-macro2, unicode-ident (Unicode-3.0 data).

## pgcore/

pyGIMLi is Copyright (C) the GIMLi development team and licensed under
Apache-2.0.

- `pgcore/src/cholmodWrapper.cpp` is derived from pyGIMLi's
  `core/src/cholmodWrapper.cpp` (Copyright (C) 2005-2024 by the GIMLi
  development team) and keeps its licence header; a notice below the header
  records the modifications.
- `pgcore/apply_rustsolve.py` and `pgcore/patches/` modify six pyGIMLi
  files: `CMakeLists.txt`, `core/src/CMakeLists.txt`, `core/src/linSolver.h`,
  `core/src/linSolver.cpp`, `core/src/cholmodWrapper.h` and
  `core/src/bert/dcfemmodelling.cpp`. The changed lines are Apache-2.0, and
  each changed file gets a modification notice.
- `pgcore/build-libgimli-msys2.sh` and `pgcore/build-libgimli-macos.sh`
  download Triangle (Jonathan R. Shewchuk) from netlib and link it into
  libgimli, as pyGIMLi's own build does. Triangle is not part of this
  repository and has its own licence: it may be distributed freely for
  non-commercial use, and distributing it as part of a commercial product
  needs its author's permission.
