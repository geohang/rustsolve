/******************************************************************************
 *   Copyright (C) 2005-2024 by the GIMLi development team                    *
 *   Carsten Rücker carsten@resistivity.net                                   *
 *                                                                            *
 *   Licensed under the Apache License, Version 2.0 (the "License");          *
 *   you may not use this file except in compliance with the License.         *
 *   You may obtain a copy of the License at                                  *
 *                                                                            *
 *       http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                            *
 *   Unless required by applicable law or agreed to in writing, software      *
 *   distributed under the License is distributed on an "AS IS" BASIS,        *
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *   See the License for the specific language governing permissions and      *
 *   limitations under the License.                                           *
 *                                                                            *
 ******************************************************************************/

/*
 * Modified 2026 by Hang Chen for rustsolve
 * (https://github.com/geohang/rustsolve): the CHOLMOD/UMFPACK calls are
 * replaced by the rustsolve C ABI (faer). The class declaration in
 * cholmodWrapper.h only gains overrides of SolverWrapper's two
 * setMatrix(const ...&) overloads, so the layout and the exported symbols the
 * Python bindings use stay the same.
 *
 * Behaviour kept from the original:
 *  - real symmetric matrices: Cholesky (was CHOLMOD), from one triangle;
 *    a matrix stored as one triangle (S.stype() != 0) is read from that
 *    triangle, the other entries ignored, as CHOLMOD did;
 *  - the symmetry test (absolute 1e-12); a full-storage matrix that fails
 *    it, or forceUmfpack / BERTUSEUMFPACK=1: LU (was UMFPACK).
 * Behaviour changed on purpose:
 *  - every failure throws (the original ignored CHOLMOD/UMFPACK status and
 *    returned whatever the solve produced), and a matrix that is refused
 *    leaves the wrapper without a factor, so a later solve throws too;
 *  - LU of a matrix stored as one triangle works on the full symmetric
 *    matrix (UMFPACK factorized the triangle alone);
 *  - stype 0 no longer means CHOLMOD's normal equations (A*A'): symmetric
 *    matrices that are not positive definite are refused (docs/design.md,
 *    open questions);
 *  - one rustsolve handle per wrapper; a new matrix with the same pattern is
 *    only refactorized, and handles share the analysis of an identical pattern
 *    (pgcore builds a new LinSolver for every wavenumber);
 *  - complex matrices are refused; rustsolve does not take over UMFPACK's
 *    complex path (docs/design.md, open questions);
 *  - the storage comes from the matrix itself (S.stype()); the stype given to
 *    the constructor is not used. The original passed it to CHOLMOD, where it
 *    chose the triangle of a full matrix, or the normal equations (0).
 */

#include "cholmodWrapper.h"
#include "vector.h"
#include "sparsematrix.h"

#include <rustsolve.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace GIMLI{

namespace {

rs_solver * handle_(void * p){ return static_cast< rs_solver * >(p); }

std::string envString_(const char * name){
    const char * v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

void check_(rs_solver * s, int32_t status, const std::string & what){
    if (status != RS_OK){
        throwError(WHERE_AM_I + " rustsolve " + what + " failed ("
                   + rs_status_name(status) + "): "
                   + (s ? rs_last_error(s) : ""));
    }
}

/*! The compressed-row arrays of S describe an n x n matrix: pointer and value
 * counts, increasing pointers, column indices in range. pgcore's constructor
 * from arrays does not check them, and the triangle conversions below index
 * with them before rustsolve sees them. */
void checkArrays_(RSparseMatrix & S){
    const std::vector < int > & cp = S.vecColPtr();
    const std::vector < int > & ri = S.vecRowIdx();
    Index n = S.rows();
    bool ok = cp.size() == n + 1 && cp[0] == 0 && cp[n] >= 0
        && (size_t)cp[n] == ri.size() && ri.size() == S.vecVals().size();
    for (Index i = 0; ok && i < n; i ++) ok = cp[i] <= cp[i + 1];
    for (size_t k = 0; ok && k < ri.size(); k ++) ok = ri[k] >= 0 && (Index)ri[k] < n;
    if (!ok) throwError(WHERE_AM_I + " inconsistent sparse matrix arrays.");
    // pgcore's rowIdx() and vals() take &v[0], which an empty array does not have
    if (ri.empty()) throwError(WHERE_AM_I + " the matrix has no entries.");
}

/*! Transpose of compressed-row arrays, as compressed rows. */
void transpose_(Index n, const std::vector < int > & cp,
                const std::vector < int > & ri, const RVector & v,
                std::vector < int > & tp, std::vector < int > & ti,
                std::vector < double > & tv){
    tp.assign(n + 1, 0);
    ti.resize(ri.size());
    tv.resize(ri.size());
    for (size_t k = 0; k < ri.size(); k ++) tp[ri[k] + 1] ++;
    for (Index i = 0; i < n; i ++) tp[i + 1] += tp[i];
    std::vector < int > next(tp.begin(), tp.end() - 1);
    for (Index i = 0; i < n; i ++){
        for (int k = cp[i]; k < cp[i + 1]; k ++){
            int d = next[ri[k]] ++;
            ti[d] = i;
            tv[d] = v[k];
        }
    }
}

/*! Full symmetric storage, as compressed rows, of a matrix stored as one
 * triangle: stype < 0 keeps columns j >= i of each row i, stype > 0 keeps
 * j <= i (pgcore's SparseMapMatrix convention); entries of the other triangle
 * are ignored, as CHOLMOD ignored them. */
void fullFromTriangle_(Index n, int stype, const std::vector < int > & cp,
                       const std::vector < int > & ri, const RVector & v,
                       std::vector < int > & fp, std::vector < int > & fi,
                       std::vector < double > & fv){
    std::vector < std::vector < std::pair < int, double > > > rows(n);
    for (Index i = 0; i < n; i ++){
        for (int k = cp[i]; k < cp[i + 1]; k ++){
            int j = ri[k];
            if (stype < 0 ? j < (int)i : j > (int)i) continue;
            rows[i].push_back(std::make_pair(j, v[k]));
            if (j != (int)i) rows[j].push_back(std::make_pair((int)i, v[k]));
        }
    }
    size_t total = 0;
    for (Index i = 0; i < n; i ++) total += rows[i].size();
    if (total > (size_t)INT_MAX){
        throwError(WHERE_AM_I + " the full matrix has too many entries for 32-bit indices.");
    }
    fp.assign(n + 1, 0);
    fi.clear();
    fv.clear();
    for (Index i = 0; i < n; i ++){
        std::sort(rows[i].begin(), rows[i].end());
        for (size_t k = 0; k < rows[i].size(); k ++){
            fi.push_back(rows[i][k].first);
            fv.push_back(rows[i][k].second);
        }
        fp[i + 1] = (int)fi.size();
    }
}

/*! Exact symmetry of a CRS matrix (both triangles stored), as the original
 * test with an absolute tolerance of 1e-12. */
bool isSymmetric_(RSparseMatrix & S){
    const std::vector < int > & cp = S.vecColPtr();
    const std::vector < int > & ri = S.vecRowIdx();
    const RVector & v = S.vecVals();
    for (Index i = 0; i < S.rows(); i ++){
        for (int j = cp[i]; j < cp[i + 1]; j ++){
            if (v[j] != 0.0){
                if (::fabs(v[j] - S.getVal(ri[j], i, false)) > 1e-12) return false;
            }
        }
    }
    return true;
}

} // namespace

bool CHOLMODWrapper::valid() { return true; }

// setMatrix releases the handle when it throws, so a constructor that throws
// (and runs no destructor) leaks nothing

CHOLMODWrapper::CHOLMODWrapper(RSparseMatrix & S, bool verbose, int stype,
                               bool forceUmfpack)
    : SolverWrapper(verbose), stype_(stype), forceUmfpack_(forceUmfpack){
    init();
    setMatrix(S);
}

CHOLMODWrapper::CHOLMODWrapper(CSparseMatrix & S, bool verbose, int stype,
                               bool forceUmfpack)
    : SolverWrapper(verbose), stype_(stype), forceUmfpack_(forceUmfpack){
    init();
    setMatrix(S);
}

CHOLMODWrapper::~CHOLMODWrapper(){
    this->free();
}

void CHOLMODWrapper::init(){
    // c_ holds the rustsolve handle; the other members of the original layout
    // stay unused
    c_ = nullptr;
    A_ = nullptr;
    L_ = nullptr;
    Numeric_ = nullptr;
    NumericD_ = nullptr;
    Ap_ = nullptr;
    Ai_ = nullptr;
    ApR_ = nullptr;
    AiR_ = nullptr;
    AxV_ = nullptr;
    AzV_ = nullptr;
    useUmfpack_ = false;
    dummy_ = false;
    name_ = "rustsolve";
}

void CHOLMODWrapper::free(){
    if (c_) rs_free(handle_(c_));
    c_ = nullptr;
}

void CHOLMODWrapper::setMatrix(RSparseMatrix & S){
    init_(S, stype_);
}

// SolverWrapper's overloads (declared by apply_rustsolve.py); S is only read
void CHOLMODWrapper::setMatrix(const RSparseMatrix & S){
    setMatrix(const_cast< RSparseMatrix & >(S));
}

void CHOLMODWrapper::setMatrix(const CSparseMatrix & S){
    setMatrix(const_cast< CSparseMatrix & >(S));
}

void CHOLMODWrapper::setMatrix(CSparseMatrix & S){
    // refused: the previous real factor must not stay usable
    this->free();
    throwError(WHERE_AM_I + " complex matrices are not supported by the "
               "rustsolve backend of this pgcore build.");
}

template < class ValueType >
void CHOLMODWrapper::init_(SparseMatrix < ValueType > & S, int stype){
    initializeMatrix_(S);
}

int CHOLMODWrapper::initializeMatrix_(CSparseMatrix & S){
    this->free();
    throwError(WHERE_AM_I + " complex matrices are not supported by the "
               "rustsolve backend of this pgcore build.");
    return 0;
}

int CHOLMODWrapper::initializeMatrix_(RSparseMatrix & S){
    // a matrix that is refused, here or by rustsolve, leaves no handle: a later
    // solve throws instead of using the previous matrix's factor
    try {
        dim_ = S.size();
        nVals_ = S.nVals();
        if (S.rows() != S.cols()){
            throwError(WHERE_AM_I + " matrix is not square: "
                       + str(S.rows()) + " x " + str(S.cols()));
        }
        if (S.rows() == 0) throwError(WHERE_AM_I + " empty matrix.");
        checkArrays_(S);

        // the same choice the original made between CHOLMOD and UMFPACK
        int stype = S.stype();
        bool lu = forceUmfpack_ || getEnvironment("BERTUSEUMFPACK", 0) == 1;
        if (!lu && stype == 0 && !isSymmetric_(S)){
            if (verbose_) log(Info, "non-symmetric matrix found .. switching to LU.");
            lu = true;
        }
        int kind = lu ? RS_KIND_LU : RS_KIND_LLT;

        // a different kind needs a new handle
        if (c_ && useUmfpack_ != lu) this->free();
        useUmfpack_ = lu;

        if (!c_){
            if (rs_abi_version() != RS_ABI_VERSION){
                throwError(WHERE_AM_I + " rustsolve library has C ABI "
                           + str(rs_abi_version()) + ", this build expects "
                           + str(RS_ABI_VERSION) + ".");
            }
            rs_options o;
            rs_default_options(&o);
            o.kind = kind;
            // pgcore stores compressed rows
            o.compressed_rows = 1;
            o.share_analysis = 1;
            // symmetry is decided above with the original's test (absolute 1e-12),
            // and a matrix that fails it goes to LU. rustsolve's own test is
            // relative and would refuse matrices the original factorized: the
            // Cholesky reads one triangle, as CHOLMOD did.
            o.check_symmetry = 0;
            // for experiments; the thread and ordering policy is an open
            // question (docs/design.md)
            std::string t = envString_("RUSTSOLVE_THREADS");
            if (!t.empty()) o.threads = std::atoi(t.c_str());
            if (envString_("RUSTSOLVE_ORDERING") == "nd") o.ordering = RS_ORDERING_ND;

            rs_solver * s = nullptr;
            check_(nullptr, rs_new(&o, &s), "init");
            c_ = s;
        }

        // With compressed rows the Cholesky reads the upper triangle of each row
        // (columns j >= i), the lower triangle of the column view. pgcore stores
        // stype < 0 as exactly that triangle, stype > 0 as the other one.
        Index n = S.rows();
        if (stype == 0 || (!lu && stype < 0)){
            check_(handle_(c_), rs_update_i32(handle_(c_), n, S.colPtr(),
                                              S.rowIdx(), S.vals()),
                   "factorization");
        } else {
            std::vector < int > tp, ti;
            std::vector < double > tv;
            if (lu){
                // LU needs both triangles
                fullFromTriangle_(n, stype, S.vecColPtr(), S.vecRowIdx(),
                                  S.vecVals(), tp, ti, tv);
            } else {
                transpose_(n, S.vecColPtr(), S.vecRowIdx(), S.vecVals(), tp, ti, tv);
            }
            check_(handle_(c_), rs_update_i32(handle_(c_), n, tp.data(), ti.data(),
                                              tv.data()),
                   "factorization");
        }

        if (verbose_){
            rs_info info;
            rs_get_info(handle_(c_), &info);
            log(Info, "rustsolve " + std::string(lu ? "LU" : "LLT")
                      + ": n=" + str(info.n) + " nnz(A)=" + str(info.nnz_a)
                      + " nnz(L)=" + str(info.nnz_factor)
                      + (info.supernodal ? " supernodal" : " simplicial")
                      + (info.reanalyzed ? (info.analysis_shared ? " analysis shared"
                                                                 : " analyzed")
                                         : " refactorized"));
        }
        return 1;
    } catch (...) {
        this->free();
        throw;
    }
}

int CHOLMODWrapper::factorise_(){
    // factorization happens in initializeMatrix_
    return 1;
}

void CHOLMODWrapper::solve(const RVector & rhs, RVector & solution){
    if (!c_) throwError(WHERE_AM_I + " no matrix set, or the last one was refused.");
    ASSERT_VEC_SIZE(rhs, this->dim_)
    solution = rhs;
    check_(handle_(c_), rs_solve(handle_(c_), 1, &solution[0], dim_), "solve");
}

void CHOLMODWrapper::solve(const CVector & rhs, CVector & solution){
    throwError(WHERE_AM_I + " complex matrices are not supported by the "
               "rustsolve backend of this pgcore build.");
}

} //namespace GIMLI
