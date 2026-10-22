"""Switch a pyGIMLi (gimli) 1.5.x source tree from CHOLMOD/UMFPACK to rustsolve.

usage: python apply_rustsolve.py <gimli source dir>

Replaces core/src/cholmodWrapper.cpp with pgcore/src/cholmodWrapper.cpp, adds
the GIMLI_RUSTSOLVE option to the two CMake files, and changes LinSolver
(linSolver.h/.cpp) and the DC forward (bert/dcfemmodelling.cpp) so that a
refused matrix leaves no solver behind and nothing leaks (LINSOLVER_* and
DCFEM_* below), and so that the primary potentials with topography are the P2
mesh's node values instead of interpolated ones (DCFEM_P2_*, DCFEM_INTERP_*).
The library's version string gets "+rustsolve"; configure a tree without git
history with -DGIMLI_VERSION=v1.5.x so the string names the source. Each
changed file gets a notice (NOTICES) under its licence header, as Apache-2.0
asks of modified files. Anchored on the lines that exist in v1.5.0-v1.5.5;
refuses to run if an anchor is missing or the tree is already patched.
"""
import pathlib
import shutil
import sys
import textwrap

HERE = pathlib.Path(__file__).resolve().parent

TOP_OLD = """find_package(UMFPACK REQUIRED)
find_package(CHOLMOD REQUIRED)
"""
TOP_NEW = """# rustsolve (faer) replaces CHOLMOD and UMFPACK in LinSolver; see
# https://github.com/geohang/rustsolve/tree/main/pgcore
option(GIMLI_RUSTSOLVE "Solve LinSolver systems with rustsolve instead of CHOLMOD/UMFPACK" ON)
if (GIMLI_RUSTSOLVE)
    find_path(RUSTSOLVE_INCLUDE_DIR rustsolve.h
              HINTS ${RUSTSOLVE_PREFIX}/include $ENV{RUSTSOLVE_PREFIX}/include)
    find_library(RUSTSOLVE_LIBRARY NAMES rustsolve_capi
                 HINTS ${RUSTSOLVE_PREFIX}/lib $ENV{RUSTSOLVE_PREFIX}/lib)
    if (NOT RUSTSOLVE_INCLUDE_DIR OR NOT RUSTSOLVE_LIBRARY)
        message(FATAL_ERROR "rustsolve not found: set RUSTSOLVE_PREFIX to a folder "
                            "with include/rustsolve.h and lib/ (rustsolve_capi)")
    endif()
    message(STATUS "rustsolve: ${RUSTSOLVE_LIBRARY} (${RUSTSOLVE_INCLUDE_DIR})")
    set(CHOLMOD_FOUND FALSE)
    set(UMFPACK_FOUND FALSE)
    # pg.core.versionStr() names the backend. GIMLI_VERSION gives the source
    # version when the tree is not a git checkout (git archive, sdist).
    if (GIMLI_VERSION)
        set(LIBGIMLI_VERSION ${GIMLI_VERSION})
    endif()
    set(PACKAGE_VERSION \\"${LIBGIMLI_VERSION}+rustsolve\\")
else()
    find_package(UMFPACK REQUIRED)
    find_package(CHOLMOD REQUIRED)
endif()
"""
CORE_OLD = """    target_link_libraries(${libgimli_TARGET_NAME} ${UMFPACK_LIBRARIES})
endif (UMFPACK_FOUND)
"""
CORE_NEW = CORE_OLD + """
if (GIMLI_RUSTSOLVE)
    include_directories(${RUSTSOLVE_INCLUDE_DIR})
    target_link_libraries(${libgimli_TARGET_NAME} ${RUSTSOLVE_LIBRARY})
endif (GIMLI_RUSTSOLVE)
"""

# LinSolver::initialize_ assigned `solver_ = new CHOLMODWrapper(...)` over the
# previous solver. When the constructor throws (rustsolve refuses the matrix),
# solver_ kept the previous matrix's solver and the next solve() used it; the
# previous solver also leaked on every setMatrix. Delete it first, and make
# solve() throw when there is no solver instead of returning zeros.
LINSOLVER_INIT_OLD = """    rows_ = S.rows();
    cols_ = S.cols();
    setSolverType(solverType_);
"""
LINSOLVER_INIT_NEW = """    // rustsolve: drop the previous matrix's solver first, so a setMatrix that
    // throws leaves none and solve() refuses instead of using the old matrix
    if (solver_) {
        delete solver_;
        solver_ = 0;
    }
""" + LINSOLVER_INIT_OLD
LINSOLVER_SOLVE_OLD = """    if (solver_) solver_->solve(rhs, solution);
"""
LINSOLVER_SOLVE_NEW = """    if (!solver_) throwError(WHERE_AM_I + " no matrix set (rustsolve: or the last one was refused)");
    solver_->solve(rhs, solution);
"""
# SolverWrapper's one-argument setMatrix(const ...&) overloads were not
# implemented by LinSolver: a call through SolverWrapper* (the DC forward with
# a solver set by the caller) or from Python with one argument threw "not yet
# implemented" and left the previous solver in place. LinSolver now overrides
# them (same slots in the vtable, so the class layout is unchanged; see
# CHOLMOD_H_* for what this does from Python).
LINSOLVER_H_OLD = """    /*! Verbose level = -1, use Linsolver.verbose(). */
    void setMatrix(CSparseMatrix & S, int stype=-2);
"""
LINSOLVER_H_NEW = LINSOLVER_H_OLD + """
    /*! rustsolve: SolverWrapper's overloads, the same as setMatrix(S, -2). */
    virtual void setMatrix(const RSparseMatrix & S);
    virtual void setMatrix(const CSparseMatrix & S);
"""
LINSOLVER_SET_OLD = """void LinSolver::setMatrix(CSparseMatrix & S, int stype){
    initialize_(S, stype);
}
"""
LINSOLVER_SET_NEW = LINSOLVER_SET_OLD + """
// rustsolve: SolverWrapper's overloads. The wrappers only read S.
void LinSolver::setMatrix(const RSparseMatrix & S){
    initialize_(const_cast< RSparseMatrix & >(S), -2);
}

void LinSolver::setMatrix(const CSparseMatrix & S){
    initialize_(const_cast< CSparseMatrix & >(S), -2);
}
"""
# The same for CHOLMODWrapper. Overriding existing virtual functions keeps the
# class layout, so the shipped bindings still load. They call SolverWrapper's
# inline versions themselves, though: from Python, these overloads take effect
# only with bindings generated from the patched headers.
CHOLMOD_H_OLD = """    virtual void setMatrix(CSparseMatrix & S);
"""
CHOLMOD_H_NEW = CHOLMOD_H_OLD + """
    /*! rustsolve: SolverWrapper's overloads, the same as the two above. */
    virtual void setMatrix(const RSparseMatrix & S);
    virtual void setMatrix(const CSparseMatrix & S);
"""
# SolverType LDL built an LDLWrapper, which without LDL (never built with
# rustsolve) prints "LDL not installed" and solves nothing: solve(b, x) left x
# as it was, solve(b) returned zeros. Refuse it instead.
LINSOLVER_LDL_OLD = """        case LDL:     solver_ = new LDLWrapper(S, verbose_); break;
"""
LINSOLVER_LDL_NEW = """        // rustsolve: no LDL in this build; LDLWrapper would solve nothing
        case LDL:     throwError(WHERE_AM_I + " SolverType LDL is not available in this build."); break;
"""
# the cached copy leaked when setMatrix threw in the constructor
LINSOLVER_MAPCTOR_OLD = """    cacheMatrix_ = new RSparseMatrix(S);

    setMatrix(dynamic_cast < RSparseMatrix &>( *cacheMatrix_));
"""
LINSOLVER_MAPCTOR_NEW = """    cacheMatrix_ = new RSparseMatrix(S);

    // rustsolve: a constructor that throws runs no destructor
    try {
        setMatrix(dynamic_cast < RSparseMatrix &>( *cacheMatrix_));
    } catch (...) {
        delete cacheMatrix_;
        cacheMatrix_ = 0;
        throw;
    }
"""
# The DC forward deleted the LinSolver it made only at the end of the
# function; a factorization or solve that throws leaked it with its factor.
DCFEM_INCLUDE_OLD = """#include "dcfemmodelling.h"
"""
DCFEM_INCLUDE_NEW = DCFEM_INCLUDE_OLD + """
#include <memory>
"""
DCFEM_DECL_OLD = """    bool solverNeedsDelete = false;
"""
DCFEM_DECL_NEW = DCFEM_DECL_OLD + """    // rustsolve: owns the LinSolver made below, also when an exception leaves
    std::unique_ptr< SolverWrapper > solverOwner;
"""
DCFEM_NEW_OLD = """        solver = new LinSolver(debug);
        solverNeedsDelete = true;
"""
DCFEM_NEW_NEW = DCFEM_NEW_OLD + """        solverOwner.reset(solver);
"""
DCFEM_DELETE_OLD = """    if (solverNeedsDelete == true){
        delete solver;
    }
"""
DCFEM_DELETE_NEW = """    if (solverNeedsDelete == true){
        solverOwner.reset();
    }
"""
# With topography the primary potentials were solved on a P2 mesh and then
# interpolated to the forward mesh's nodes. interpolate() takes, at each node,
# the first cell around it in std::set<Cell*> order (heap addresses), and on
# macOS arm64 TriangleShape::xyz2rst is compiled to fused multiply-adds, so at
# a node the shape functions are 0 and 1 only to rounding: the value then
# depended on the process (up to 2.8e-16 of the largest value, which the
# 1e7-contrast case turns into 5e-13 in a response). The P2 mesh made here
# starts with the forward mesh's nodes (Mesh::createRefined_ copies them first,
# in order), so take their values. On x86-64 builds the interpolation was exact
# at the nodes and both give the same numbers. A primary mesh given by the
# caller (setPrimaryMesh) is still interpolated.
DCFEM_P2_OLD = """                if (!primMesh_){
                    primMesh_ = new Mesh;
"""
DCFEM_P2_NEW = """                // rustsolve: a P2 mesh made here starts with the forward mesh's
                // nodes, and its values there are taken below
                bool primMeshMadeHere = !primMesh_;
""" + DCFEM_P2_OLD
DCFEM_INTERP_OLD = """                interpolate(*primMesh_, primPotentials, mesh_->positions(), *primPot_, verbose_);
"""
DCFEM_INTERP_NEW = """                if (primMeshMadeHere){
                    // rustsolve: node j of the forward mesh is node j of the P2
                    // mesh (Mesh::createRefined_), so take its value instead of
                    // interpolating it: interpolate() takes the first cell around
                    // a node in heap-address order, and where the shape functions
                    // are 0 and 1 only to rounding (fused multiply-adds on macOS
                    // arm64) the value changed from one process to the next
                    Index n = mesh_->nodeCount();
                    if (primMesh_->nodeCount() < n){
                        throwError(WHERE_AM_I + " the P2 mesh has fewer nodes than the forward mesh.");
                    }
                    for (Index j = 0; j < n; j ++){
                        if (primMesh_->node(j).pos() != mesh_->node(j).pos()){
                            throwError(WHERE_AM_I + " node " + str(j) + " of the P2 mesh is not "
                                       "the forward mesh's node " + str(j) + ".");
                        }
                    }
                    if (primPot_->rows() != primPotentials.rows()){
                        primPot_->resize(primPotentials.rows(), n);
                    }
                    for (Index i = 0; i < primPotentials.rows(); i ++){
                        if (primPotentials[i].size() == 0) continue;
                        if (primPotentials[i].size() != primMesh_->nodeCount()){
                            throwLengthError(WHERE_AM_I + " primary potential " + str(i) + " has "
                                             + str(primPotentials[i].size()) + " values, the P2 mesh "
                                             + str(primMesh_->nodeCount()) + " nodes.");
                        }
                        for (Index j = 0; j < n; j ++) (*primPot_)[i][j] = primPotentials[i][j];
                    }
                } else {
                    interpolate(*primMesh_, primPotentials, mesh_->positions(), *primPot_, verbose_);
                }
"""


NOTICE_PREFIX = "Modified 2026 by Hang Chen for rustsolve (https://github.com/geohang/rustsolve): "
NOTICES = {
    "CMakeLists.txt": "option GIMLI_RUSTSOLVE links rustsolve in place of UMFPACK and CHOLMOD.",
    "core/src/CMakeLists.txt": "links rustsolve_capi when GIMLI_RUSTSOLVE is on.",
    "core/src/linSolver.cpp": "a refused matrix leaves no solver behind, SolverWrapper's "
                              "one-argument setMatrix overloads, SolverType LDL is refused.",
    "core/src/linSolver.h": "declares SolverWrapper's one-argument setMatrix overloads.",
    "core/src/cholmodWrapper.h": "declares SolverWrapper's one-argument setMatrix overloads.",
    "core/src/bert/dcfemmodelling.cpp": "the LinSolver made here is owned by a std::unique_ptr; "
                                        "with topography the primary potentials are the P2 "
                                        "mesh's node values.",
}


def add_notice(text, notice):
    """Put the notice under a C++ file's licence header, or atop a CMake file."""
    if text.startswith("/*"):
        end = text.index("\n", text.index("*/") + 2) + 1
        body = "".join(f" * {line}\n" for line in textwrap.wrap(notice, 74, break_on_hyphens=False))
        return text[:end] + "\n/*\n" + body + " */\n" + text[end:]
    return "".join(f"# {line}\n" for line in textwrap.wrap(notice, 76, break_on_hyphens=False)) + text


def edit(path, pairs, marker="GIMLI_RUSTSOLVE", notice=None):
    """Replace each (old, new, count) anchor, which must occur exactly count times."""
    raw = path.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    text = raw.replace("\r\n", "\n")
    if marker in text:
        sys.exit(f"{path} is already patched")
    for old, new, count in pairs:
        if text.count(old) != count:
            sys.exit(f"{path}: anchor found {text.count(old)} times, expected {count}")
        text = text.replace(old, new)
    if notice:
        text = add_notice(text, NOTICE_PREFIX + NOTICES[notice])
    path.write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))


def main():
    src = pathlib.Path(sys.argv[1]).resolve()
    if not (src / "core" / "src" / "cholmodWrapper.h").exists():
        sys.exit(f"{src} is not a gimli source tree")
    edit(src / "CMakeLists.txt", [(TOP_OLD, TOP_NEW, 1)], notice="CMakeLists.txt")
    edit(src / "core" / "src" / "CMakeLists.txt", [(CORE_OLD, CORE_NEW, 1)],
         notice="core/src/CMakeLists.txt")
    edit(src / "core" / "src" / "linSolver.cpp",
         [(LINSOLVER_INIT_OLD, LINSOLVER_INIT_NEW, 2), (LINSOLVER_SOLVE_OLD, LINSOLVER_SOLVE_NEW, 4),
          (LINSOLVER_SET_OLD, LINSOLVER_SET_NEW, 1), (LINSOLVER_MAPCTOR_OLD, LINSOLVER_MAPCTOR_NEW, 1),
          (LINSOLVER_LDL_OLD, LINSOLVER_LDL_NEW, 2)],
         marker="rustsolve", notice="core/src/linSolver.cpp")
    edit(src / "core" / "src" / "linSolver.h", [(LINSOLVER_H_OLD, LINSOLVER_H_NEW, 1)],
         marker="rustsolve", notice="core/src/linSolver.h")
    edit(src / "core" / "src" / "cholmodWrapper.h", [(CHOLMOD_H_OLD, CHOLMOD_H_NEW, 1)],
         marker="rustsolve", notice="core/src/cholmodWrapper.h")
    edit(src / "core" / "src" / "bert" / "dcfemmodelling.cpp",
         [(DCFEM_INCLUDE_OLD, DCFEM_INCLUDE_NEW, 1), (DCFEM_DECL_OLD, DCFEM_DECL_NEW, 2),
          (DCFEM_NEW_OLD, DCFEM_NEW_NEW, 2), (DCFEM_DELETE_OLD, DCFEM_DELETE_NEW, 2),
          (DCFEM_P2_OLD, DCFEM_P2_NEW, 1), (DCFEM_INTERP_OLD, DCFEM_INTERP_NEW, 1)],
         marker="rustsolve", notice="core/src/bert/dcfemmodelling.cpp")
    shutil.copyfile(HERE / "src" / "cholmodWrapper.cpp", src / "core" / "src" / "cholmodWrapper.cpp")
    print(f"patched {src}")


if __name__ == "__main__":
    main()
