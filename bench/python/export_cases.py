"""Export representative ERT systems from PyHydroGeophysX's own workflows.

usage: python export_cases.py <case> <outroot>
Run each case under run_capped.py (separate process, RSS cap).
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import pygimli as pg
from pygimli.physics import ert

import capture
import caseio

import paths

DATA = str(paths.DATA)


def _write_log(outdir, cap, extra=None):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "capture_log.json"), "w") as f:
        json.dump(dict(log=cap.log, saved=cap.saved, **(extra or {})), f, indent=1)


def _inversion(case, outroot, data_file, **kw):
    """The package's quick-start inversion, every forward solve captured."""
    from PyHydroGeophysX.inversion.ert_inversion import run_ert_manager_inversion
    outdir = os.path.join(outroot, case)
    sel = lambda label, k: label == "resp01" or k == 0
    cap = capture.Capture(outdir, sel, meta=dict(workflow=f"run_ert_manager_inversion({os.path.basename(data_file)})"))
    capture.install(cap)
    t0 = time.perf_counter()
    res = run_ert_manager_inversion(data_file, os.path.join(outroot, "_inv_" + case), **kw)
    cap.finish()
    _write_log(outdir, cap, dict(chi2=float(res.get("chi2", np.nan)), engine=res.get("engine"),
                                 wall=time.perf_counter() - t0))
    print(case, "saved", len(cap.saved), "matrices", len(cap.log), "chi2", res.get("chi2"))


def line2(outroot):
    _inversion("line2", outroot, os.path.join(DATA, "ERT", "Bert", "fielddataline2.dat"),
               max_iterations=4)


def e4d(outroot):
    _inversion("e4d", outroot, os.path.join(DATA, "ERT", "E4D", "2021-10-08_1400.ohm"),
               max_iterations=3)


def _forward(cap, label, fop, model, **extra):
    cap.begin(label, **extra)
    r = np.array(fop.response(model), dtype=float)
    cap.finish()
    return r


def tl_synth(outroot):
    """Synthetic one-year time lapse: the same forward mesh, the model of four days."""
    outdir = os.path.join(outroot, "tl_synth")
    d = os.path.join(DATA, "TL_measurements")
    mesh = pg.load(os.path.join(d, "mesh.bms"))
    res = np.load(os.path.join(d, "resistivity_mesh.npy"), mmap_mode="r")
    scheme = ert.load(os.path.join(d, "appres", "synthetic_data120.dat"))
    mesh.setCellMarkers(np.ones(mesh.cellCount()) * 2)
    grid = pg.meshtools.appendTriangleBoundary(mesh, marker=1, xbound=100, ybound=100)
    cap = capture.Capture(outdir, lambda label, k: k in (0, 7), meta=dict(
        workflow="examples/Ex_Time_lapse_measurement.py forward (ERTModelling, region manager)"))
    fop = ert.ERTModelling()
    fop.setData(scheme)
    fop.setMesh(grid)
    fop._core.setSolver(cap)
    for day in (30, 150, 210, 330):
        m = np.array(res[day], dtype=float)
        _forward(cap, f"day{day:03d}", fop, m, rho_min=float(m.min()), rho_max=float(m.max()))
    _write_log(outdir, cap, dict(mesh=str(grid)))


def _para_mesh(data_file):
    from PyHydroGeophysX.inversion.ert_mesh import build_inversion_mesh
    data = ert.load(data_file)
    # analytic k: it scales the response only, the matrices do not depend on it
    data["k"] = ert.createGeometricFactors(data, numerical=False)
    mesh = build_inversion_mesh(data)
    return data, mesh


def contrast(outroot):
    """High conductivity contrast on the field line's own inversion mesh.

    Host 1e4 ohm-m. Cases: a 1 ohm-m body (contrast 1e4, the package's
    default rho bounds), a 1e-3 ohm-m body (1e7, a metal pipe or brine), and a
    checkerboard of 1 and 1e4 ohm-m alternating cells.
    """
    outdir = os.path.join(outroot, "contrast")
    data, mesh = _para_mesh(os.path.join(DATA, "ERT", "Bert", "fielddataline2.dat"))
    cap = capture.Capture(outdir, lambda label, k: k in (0, 7), meta=dict(
        workflow="ERTModelling on build_inversion_mesh(fielddataline2), region manager (H2, SR)"))
    fop = ert.ERTModelling()
    fop.setData(data)
    fop.setMesh(mesh)
    fop._core.setSolver(cap)
    pd = fop.paraDomain
    c = np.array([[cc.center().x(), cc.center().y()] for cc in pd.cells()])
    xs = np.array([p.x() for p in data.sensors()])
    ys = np.array([p.y() for p in data.sensors()])
    x0 = 0.5 * (xs.min() + xs.max())
    y0 = np.interp(x0, xs, ys) - 6.0
    body = (c[:, 0] - x0) ** 2 + (c[:, 1] - y0) ** 2 < 4.0 ** 2
    for name, rb in (("body1", 1.0), ("body1e-3", 1e-3)):
        m = np.full(pd.cellCount(), 1e4)
        m[body] = rb
        _forward(cap, name, fop, m, host=1e4, body=rb, body_cells=int(body.sum()))
    m = np.where(np.arange(pd.cellCount()) % 2 == 0, 1.0, 1e4)
    _forward(cap, "checker", fop, m, rho_min=1.0, rho_max=1e4)
    _write_log(outdir, cap, dict(para_cells=pd.cellCount()))


def p2_primary(outroot):
    """Topography: the primary potential pgcore solves on the P2 forward mesh."""
    outdir = os.path.join(outroot, "p2_primary")
    data, mesh = _para_mesh(os.path.join(DATA, "ERT", "Bert", "fielddataline2.dat"))
    m2 = mesh.createP2()
    cap = capture.Capture(outdir, lambda label, k: k in (0, 7), meta=dict(
        workflow="ERTModelling(sr=False) on build_inversion_mesh(fielddataline2).createP2(), homogeneous"))
    fop = ert.ERTModelling(sr=False)
    fop.setData(data)
    fop.setMesh(m2, ignoreRegionManager=True)
    fop._core.setSolver(cap)
    _forward(cap, "p2hom", fop, np.full(m2.cellCount(), 100.0), electrode_nodes=capture.electrode_nodes(m2, data))
    _write_log(outdir, cap, dict(mesh=str(m2)))


def grid3d(outroot):
    """A 3D tetrahedral forward mesh (one solve per electrode, no wavenumbers)."""
    outdir = os.path.join(outroot, "grid3d")
    x = np.concatenate([[-60, -40, -25, -15], np.arange(-10, 10.01, 0.75), [15, 25, 40, 60]])
    z = -np.concatenate([np.arange(0, 8.01, 1.0), [10, 13, 17, 23, 32, 45, 60]])[::-1]
    g = pg.createGrid(x, x, z)
    mesh = pg.meshtools.refineHex2Tet(g)
    for b in mesh.boundaries():
        if b.outside():
            b.setMarker(pg.core.MARKER_BOUND_HOMOGEN_NEUMANN if abs(b.center().z()) < 1e-9
                        else pg.core.MARKER_BOUND_MIXED)
    ex = np.array([-7.0, -4.0, -1.0, 2.0, 5.0])
    elecs = [[xi, yi, 0.0] for yi in ex for xi in ex]
    data = pg.DataContainerERT()
    for e in elecs:
        data.createSensor(e)
    ne = len(elecs)
    for i in range(ne - 3):
        data.createFourPointData(data.size(), i, i + 1, i + 3, i + 2)
    data["k"] = ert.createGeometricFactors(data, numerical=False)
    cap = capture.Capture(outdir, lambda label, k: True, meta=dict(
        workflow="ERTModelling 3D on createGrid+refineHex2Tet"))
    fop = ert.ERTModelling()
    fop.setData(data)
    fop.setMesh(mesh, ignoreRegionManager=True)
    fop._core.setSolver(cap)
    c = np.array([[cc.center().x(), cc.center().y(), cc.center().z()] for cc in mesh.cells()])
    m = np.full(mesh.cellCount(), 100.0)
    m[(np.abs(c[:, 0]) < 3) & (np.abs(c[:, 1]) < 3) & (c[:, 2] > -5) & (c[:, 2] < -2)] = 0.1
    _forward(cap, "body3d", fop, m, electrode_nodes=capture.electrode_nodes(mesh, data),
             rho_min=0.1, rho_max=100.0)
    _write_log(outdir, cap, dict(mesh=str(mesh)))


def failing(outroot):
    """Systems a solver must refuse or flag, from the same forward path.

    neg:   a model with negative resistivity cells (an indefinite matrix);
    nan:   a model with a NaN cell;
    inf:   a block of infinite resistivity (zero conductivity): nodes inside it
           have empty rows, the matrix is singular;
    neumann: the pure-Neumann P1 stiffness matrix of the same mesh without any
           Dirichlet or mixed boundary (k = 0): constant null space; RHS a
           consistent dipole and an inconsistent pole.
    """
    outdir = os.path.join(outroot, "failing")
    data, mesh = _para_mesh(os.path.join(DATA, "ERT", "Bert", "fielddataline2.dat"))
    cap = capture.Capture(outdir, lambda label, k: k == 0, meta=dict(
        workflow="ERTModelling on build_inversion_mesh(fielddataline2), region manager"))
    fop = ert.ERTModelling()
    fop.setData(data)
    fop.setMesh(mesh)
    fop._core.setSolver(cap)
    pd = fop.paraDomain
    nc = pd.cellCount()
    c = np.array([[cc.center().x(), cc.center().y()] for cc in pd.cells()])
    out = {}
    for name in ("neg", "nan", "inf"):
        m = np.full(nc, 100.0)
        if name == "neg":
            m[np.argsort(c[:, 1])[: nc // 20]] = -50.0
        elif name == "nan":
            m[nc // 2] = np.nan
        else:
            xm = np.median(c[:, 0])
            ym = np.min(c[:, 1]) + 0.3 * (np.max(c[:, 1]) - np.min(c[:, 1]))
            m[(np.abs(c[:, 0] - xm) < 15) & (np.abs(c[:, 1] - ym) < 6)] = np.inf
        try:
            r = _forward(cap, name, fop, m)
            out[name] = dict(finite=int(np.isfinite(r).sum()), n=int(r.size),
                             rmin=float(np.nanmin(r)) if np.isfinite(r).any() else None,
                             rmax=float(np.nanmax(r)) if np.isfinite(r).any() else None)
        except Exception as e:  # pgcore may raise; record it
            cap.finish()
            out[name] = dict(error=repr(e))
        print(name, out[name], flush=True)
    # pure Neumann stiffness of the same para mesh (P1, sigma = 1/100)
    A = pg.solver.createStiffnessMatrix(pd, a=pg.Vector(nc, 0.01))
    S = pg.matrix.SparseMatrix(A)
    ip = np.array(S.vecColPtr(), dtype=np.int64)
    ii = np.array(S.vecRowIdx(), dtype=np.int64)
    vv = np.array(S.vecVals(), dtype=float)
    n = S.rows()
    en = capture.electrode_nodes(pd, data)
    b = np.zeros((n, 2))
    b[en[5], 0], b[en[30], 0] = 1.0, -1.0     # dipole: consistent
    b[en[5], 1] = 1.0                          # pole: inconsistent
    ref = pg.core.LinSolver(False)
    ref.setMatrix(S, 1)
    x = np.column_stack([np.array(ref.solve(pg.Vector(b[:, j])), dtype=float) for j in range(2)])
    caseio.save_case(os.path.join(outdir, "neumann_k00"), ip, ii, vv, b, x,
                     dict(workflow="pg.solver.createStiffnessMatrix(paraDomain), no BC",
                          label="neumann", ref_solver=ref.solverName(), electrode_nodes=en))
    _write_log(outdir, cap, dict(results=out))



def _scale3d(outroot, h, tag):
    """Scaling case: P1 stiffness of a 3D tetrahedral grid (pyGIMLi assembly),
    homogeneous Dirichlet on the far sides and bottom, a conductive body.
    Not the DC forward path (no mixed boundary, no SR): it is here to see how
    the backends scale on 3D fill, where the forward is solver-bound."""
    import scipy.sparse as sp
    outdir = os.path.join(outroot, "scale3d")
    x = np.concatenate([[-60, -40, -25, -15], np.arange(-10, 10.0001, h), [15, 25, 40, 60]])
    zz = -np.concatenate([np.arange(0, 8.0001, h), [10, 13, 17, 23, 32, 45, 60]])[::-1]
    g = pg.createGrid(x, x, zz)
    mesh = pg.meshtools.refineHex2Tet(g)
    del g
    # jitter interior nodes (15 % of the local spacing) so the tetrahedra are
    # generic: on the regular Kuhn split the face-diagonal couplings are exactly
    # zero and the matrix would collapse to a 7-point stencil
    rng = np.random.default_rng(7)
    P0 = np.array(mesh.positions())
    lim = 59.999
    inner = (np.abs(P0[:, 0]) < lim) & (np.abs(P0[:, 1]) < lim) & (P0[:, 2] > -lim) & (P0[:, 2] < -1e-9)
    for i in np.flatnonzero(inner):
        hx = h if abs(P0[i, 0]) < 10 else 2.0
        hz = h if P0[i, 2] > -8 else 1.0
        d = rng.uniform(-0.15, 0.15, 3) * np.array([hx, hx, hz])
        mesh.node(int(i)).setPos(pg.Pos(*(P0[i] + d)))
    print(mesh, "min cell size", float(np.min(np.array([c.size() for c in mesh.cells()]))), flush=True)
    c = np.array(mesh.cellCenters())
    rho = np.full(mesh.cellCount(), 100.0)
    rho[(np.abs(c[:, 0]) < 3) & (np.abs(c[:, 1]) < 3) & (c[:, 2] > -5) & (c[:, 2] < -2)] = 0.1
    A = pg.solver.createStiffnessMatrix(mesh, a=pg.Vector(1.0 / rho))
    S = pg.matrix.SparseMatrix(A)
    del A
    n = S.rows()
    M = sp.csr_matrix((np.array(S.vecVals(), dtype=float), np.array(S.vecRowIdx(), dtype=np.int64),
                       np.array(S.vecColPtr(), dtype=np.int64)), shape=(n, n))
    del S
    P = np.array(mesh.positions())
    far = (np.abs(np.abs(P[:, 0]) - 60) < 1e-9) | (np.abs(np.abs(P[:, 1]) - 60) < 1e-9) | (np.abs(P[:, 2] + 60) < 1e-9)
    # homogeneous Dirichlet as pgcore applies it: clean row and column, keep
    # the pattern (explicit zeros), unit diagonal
    M = M.tocoo()
    hit = far[M.row] | far[M.col]
    M.data[hit] = 0.0
    M.data[(M.row == M.col) & far[M.row]] = 1.0
    M = M.tocsr()
    M.sort_indices()
    ex = np.array([-7.0, -4.0, -1.0, 2.0, 5.0])
    en = [int(mesh.findNearestNode([xi, yi, 0.0])) for yi in ex for xi in ex]
    del mesh
    B = np.zeros((n, len(en)))
    for j, k in enumerate(en):
        B[k, j] = 1.0
    S2 = pg.matrix.SparseMatrix(pg.core.IndexArray(M.indptr.astype(np.int64)),
                                pg.core.IndexArray(M.indices.astype(np.int64)), pg.Vector(M.data))
    t0 = time.perf_counter()
    ref = pg.core.LinSolver(False)
    ref.setMatrix(S2, 1)
    tf = time.perf_counter() - t0
    X = np.column_stack([np.array(ref.solve(pg.Vector(B[:, j])), dtype=float) for j in range(B.shape[1])])
    caseio.save_case(os.path.join(outdir, tag), M.indptr, M.indices, M.data, B, X,
                     dict(workflow=f"pg.solver.createStiffnessMatrix(3D tet grid h={h}) + Dirichlet far sides",
                          label=tag, ref_solver=ref.solverName(), ref_tfact=tf, electrode_nodes=en))
    print(tag, n, M.nnz, "cholmod factor", tf, flush=True)


def scale3d_57k(outroot):
    _scale3d(outroot, 0.5, "h050")


def scale3d_97k(outroot):
    _scale3d(outroot, 0.4, "h040")


if __name__ == "__main__":
    case, outroot = sys.argv[1], sys.argv[2]
    os.makedirs(outroot, exist_ok=True)
    globals()[case](outroot)
