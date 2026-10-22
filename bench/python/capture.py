"""Capture the systems pgcore's ERT forward solves, without changing its path.

``Capture`` is a ``pg.core.SolverWrapper`` set on a DC modelling core with
``setSolver``. For every matrix pgcore assembles (one per 2.5D wavenumber and
response) it builds the same reference pgcore builds itself - a
``pg.core.LinSolver`` with ``setMatrix(S, 1)``, i.e. CHOLMOD on the upper
triangle - and solves every right-hand side with it, so the forward results
are pgcore's own. Selected systems are written with ``caseio.save_case``.
"""
import time

import numpy as np
import pygimli as pg
from pygimli.physics import ert

import caseio


class Capture(pg.core.SolverWrapper):
    def __init__(self, outdir, select, meta=None):
        super().__init__(False)
        self.outdir = outdir
        self.select = select          # select(label, k_idx) -> bool
        self.meta = dict(meta or {})
        self.label = "resp0"
        self.k_idx = -1
        self.cur = None
        self.ref = None
        self.Sc = None
        self.log = []                 # per-matrix timings of the reference
        self.saved = []

    def begin(self, label, **extra):
        self._flush()
        self.label = label
        self.k_idx = -1
        self.extra = extra

    def setMatrix(self, S):
        self._flush()
        self.k_idx += 1
        self.Sc = pg.matrix.SparseMatrix(S)
        t0 = time.perf_counter()
        self.ref = pg.core.LinSolver(False)
        self.ref.setMatrix(self.Sc, 1)
        tf = time.perf_counter() - t0
        save = bool(self.select(self.label, self.k_idx))
        self.cur = dict(save=save, tfact=tf, tsolve=0.0, nsolve=0,
                        n=int(S.rows()), nnz=int(S.nVals()),
                        solver=self.ref.solverName(), rhs=[], x=[])
        if save:
            self.cur["A"] = (np.array(S.vecColPtr(), dtype=np.int64),
                             np.array(S.vecRowIdx(), dtype=np.int64),
                             np.array(S.vecVals(), dtype=float))

    def solve(self, rhs, sol):
        r = pg.Vector(rhs)
        t0 = time.perf_counter()
        x = self.ref.solve(r)
        self.cur["tsolve"] += time.perf_counter() - t0
        self.cur["nsolve"] += 1
        sol.setVal(x, 0, len(x))
        if self.cur["save"]:
            self.cur["rhs"].append(np.array(r, dtype=float))
            self.cur["x"].append(np.array(x, dtype=float))

    def _flush(self):
        c, self.cur = self.cur, None
        if c is None:
            return
        self.log.append(dict(label=self.label, k=self.k_idx, n=c["n"], nnz=c["nnz"],
                             solver=c["solver"], tfact=c["tfact"],
                             tsolve=c["tsolve"], nsolve=c["nsolve"]))
        if c["save"] and c["nsolve"]:
            name = f"{self.label}_k{self.k_idx:02d}"
            meta = dict(self.meta)
            meta.update(getattr(self, "extra", {}) or {})
            meta.update(label=self.label, k_idx=self.k_idx,
                        ref_solver=c["solver"], ref_tfact=c["tfact"],
                        ref_tsolve=c["tsolve"])
            caseio.save_case(f"{self.outdir}/{name}", *c["A"],
                             np.column_stack(c["rhs"]), np.column_stack(c["x"]),
                             meta)
            self.saved.append(name)

    def finish(self):
        self._flush()


def electrode_nodes(mesh, data):
    return [int(mesh.findNearestNode(p)) for p in data.sensors()]


_INSTALLED = {}


def install(cap):
    """Make every ert.ERTModelling built from now on solve through ``cap``."""
    cls = ert.ERTModelling
    if "init" not in _INSTALLED:
        _INSTALLED["init"] = cls.__init__
        _INSTALLED["response"] = cls.response
        _INSTALLED["jac"] = cls.createJacobian
        _INSTALLED["count"] = [0, 0]

    def __init__(self, *a, **k):
        _INSTALLED["init"](self, *a, **k)
        self._core.setSolver(_INSTALLED["cap"])

    def response(self, mod):
        _INSTALLED["count"][0] += 1
        _INSTALLED["cap"].begin(f"resp{_INSTALLED['count'][0]:02d}")
        return _INSTALLED["response"](self, mod)

    def createJacobian(self, mod):
        _INSTALLED["count"][1] += 1
        _INSTALLED["cap"].begin(f"jac{_INSTALLED['count'][1]:02d}")
        return _INSTALLED["jac"](self, mod)

    _INSTALLED["cap"] = cap
    cls.__init__ = __init__
    cls.response = response
    cls.createJacobian = createJacobian
