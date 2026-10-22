// Checks of the rustsolve pgcore patch that Python cannot reach with the
// shipped bindings: SolverWrapper's four one-argument setMatrix overrides (the
// DC forward's call through SolverWrapper*), a const matrix left unchanged,
// the refusal of a complex matrix, of SolverType LDL and of a matrix without
// entries, and that no refused matrix leaves a solver behind. A refusal must
// carry the patch's own message, so a refusal from somewhere else (an
// unpatched path that fails later) does not count. Exits 0 when every check
// passes. Build and run: run-check-msys2.sh (Windows) or run-check-macos.sh.
#include <cholmodWrapper.h>
#include <linSolver.h>
#include <sparsematrix.h>
#include <vector.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace GIMLI;

static int failures = 0;

static void check(bool ok, const std::string & what){
    std::cout << (ok ? "ok   " : "FAIL ") << what << std::endl;
    if (!ok) failures ++;
}

//! true when f throws a std::exception whose message contains part
template < class F > static bool throws(F f, const std::string & part){
    try {
        f();
    } catch (std::exception & e) {
        if (std::string(e.what()).find(part) != std::string::npos) return true;
        std::cout << "      (threw: " << e.what() << ")" << std::endl;
        return false;
    } catch (...) {
        std::cout << "      (threw a non-std exception)" << std::endl;
        return false;
    }
    return false;
}

// [[a, b], [b, c]] in full storage
static RSparseMatrix full2(double a, double b, double c){
    std::vector < int > cp = {0, 2, 4}, ri = {0, 1, 0, 1};
    RVector v(4);
    v[0] = a; v[1] = b; v[2] = b; v[3] = c;
    return RSparseMatrix(cp, ri, v, 0);
}

static CSparseMatrix complex2(){
    CSparseMapMatrix M(2, 2);
    M.setVal(0, 0, Complex(2, 0)); M.setVal(0, 1, Complex(1, 0));
    M.setVal(1, 0, Complex(1, 0)); M.setVal(1, 1, Complex(3, 0));
    return CSparseMatrix(M);
}

static bool near(const RVector & x, double a, double b){
    return std::fabs(x[0] - a) < 1e-14 && std::fabs(x[1] - b) < 1e-14;
}

// what the patch's messages contain
static const std::string NO_SOLVER = "no matrix set";
static const std::string COMPLEX = "complex matrices are not supported";
static const std::string NOT_PD = "NOT_POSITIVE_DEFINITE";
static const std::string NO_LDL = "SolverType LDL is not available";

int main(){
    RVector b(2);
    b[0] = 3.0; b[1] = 4.0;
    RSparseMatrix A = full2(2, 1, 3);          // A x = b: x = [1, 1]
    const RSparseMatrix B = full2(4, 1, 5);    // x = [11/19, 13/19]
    const RSparseMatrix N = full2(-2, 1, -3);  // negative definite: refused
    const CSparseMatrix C = complex2();
    const double b0 = 11.0 / 19, b1 = 13.0 / 19;

    // LinSolver::setMatrix(const RSparseMatrix &)
    try {
        LinSolver ls(A);
        SolverWrapper * w = &ls;
        std::vector < double > vals;
        std::vector < int > rows, cols;
        for (Index i = 0; i < B.vecVals().size(); i ++) vals.push_back(B.vecVals()[i]);
        for (size_t i = 0; i < B.vecRowIdx().size(); i ++) rows.push_back(B.vecRowIdx()[i]);
        for (size_t i = 0; i < B.vecColPtr().size(); i ++) cols.push_back(B.vecColPtr()[i]);
        w->setMatrix(B);
        RVector x(2);
        ls.solve(b, x);
        check(near(x, b0, b1), "LinSolver through SolverWrapper::setMatrix(const RSparseMatrix &)");
        bool same = B.rows() == 2 && B.cols() == 2 && B.stype() == 0
            && rows.size() == B.vecRowIdx().size() && cols.size() == B.vecColPtr().size();
        for (size_t i = 0; same && i < vals.size(); i ++) same = vals[i] == B.vecVals()[i];
        for (size_t i = 0; same && i < rows.size(); i ++) same = rows[i] == B.vecRowIdx()[i];
        for (size_t i = 0; same && i < cols.size(); i ++) same = cols[i] == B.vecColPtr()[i];
        check(same, "the const matrix is left unchanged (values, indices, pointers, shape, stype)");
        check(throws([&]{ w->setMatrix(N); }, NOT_PD), "a refused matrix through SolverWrapper * throws");
        check(throws([&]{ RVector y(2); ls.solve(b, y); }, NO_SOLVER), "the solve after the refusal throws");
        w->setMatrix(B);
        RVector z(2);
        ls.solve(b, z);
        check(near(z, b0, b1), "a later matrix is accepted again");
    } catch (std::exception & e) {
        check(false, std::string("LinSolver, real: unexpected exception: ") + e.what());
    }
    // LinSolver::setMatrix(const CSparseMatrix &)
    try {
        LinSolver ls(A);
        SolverWrapper * w = &ls;
        check(throws([&]{ w->setMatrix(C); }, COMPLEX), "LinSolver: a complex matrix through SolverWrapper * is refused");
        check(throws([&]{ RVector y(2); ls.solve(b, y); }, NO_SOLVER), "LinSolver: the real solve after it throws");
    } catch (std::exception & e) {
        check(false, std::string("LinSolver, complex: unexpected exception: ") + e.what());
    }
    // CHOLMODWrapper::setMatrix(const RSparseMatrix &) and (const CSparseMatrix &)
    try {
        CHOLMODWrapper c(A, false, 0, false);
        SolverWrapper * w = &c;
        RVector x(2);
        c.solve(b, x);
        check(near(x, 1, 1), "CHOLMODWrapper solves A");
        w->setMatrix(B);
        RVector y(2);
        c.solve(b, y);
        check(near(y, b0, b1), "CHOLMODWrapper through SolverWrapper::setMatrix(const RSparseMatrix &)");
        check(throws([&]{ w->setMatrix(C); }, COMPLEX), "CHOLMODWrapper: a complex matrix through SolverWrapper * is refused");
        check(throws([&]{ RVector z(2); c.solve(b, z); }, NO_SOLVER), "CHOLMODWrapper: the real solve after it throws");
    } catch (std::exception & e) {
        check(false, std::string("CHOLMODWrapper: unexpected exception: ") + e.what());
    }
    // SolverType LDL: the constructors, and a switch on a solver that has a factor
    check(throws([&]{ LinSolver l(A, LDL); }, NO_LDL), "SolverType LDL is refused (real)");
    check(throws([&]{ CSparseMatrix Cc = complex2(); LinSolver l(Cc, LDL); }, NO_LDL), "SolverType LDL is refused (complex)");
    try {
        LinSolver ls(A);
        ls.setSolverType(LDL);
        RSparseMatrix Bc = B;
        check(throws([&]{ ls.setMatrix(Bc); }, NO_LDL), "LDL after a factor: setMatrix is refused");
        check(throws([&]{ RVector y(2); ls.solve(b, y); }, NO_SOLVER), "LDL after a factor: the solve after it throws");
    } catch (std::exception & e) {
        check(false, std::string("LDL after a factor: unexpected exception: ") + e.what());
    }
    // a matrix without entries, in each storage, and one with no rows (from map
    // matrices: pgcore's constructor from arrays takes max() of the row indices)
    for (int stype : {0, -1, 1}){
        RSparseMapMatrix M(2, 2, stype);
        RSparseMatrix E(M);
        check(E.stype() == stype && E.rows() == 2 && throws([&]{ LinSolver l(E); }, "the matrix has no entries"),
              "a matrix without entries is refused, stype " + std::to_string(stype));
    }
    {
        RSparseMapMatrix M(0, 0);
        RSparseMatrix Z(M);
        check(throws([&]{ LinSolver l(Z); }, "empty matrix"), "a matrix with no rows is refused");
    }
    std::cout << (failures ? "FAILED" : "all checks passed") << std::endl;
    return failures ? 1 : 0;
}
