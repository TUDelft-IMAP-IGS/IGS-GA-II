"""
benchmark_dascmop.py
====================
DAS-CMOP benchmark suite comparing IMAP-GA against state-of-the-art
multi-objective and single-objective algorithms.

Problems
--------
DAS-CMOP1 - DAS-CMOP6   bi-objective, 30 vars, 11 constraints  (difficulty 1-16)
DAS-CMOP7 - DAS-CMOP9   tri-objective, 30 vars,  7 constraints  (difficulty_factors=[0.5,0.5,0.5])

Reference: Liu et al. 2019, "Indicator-based constrained multiobjective
evolutionary algorithms", IEEE TSMC.

Algorithms compared
-------------------
IMAP-GA   : NSGA-II backbone with IMAP affine-aggregation survival (this work).
              Objective bounds estimated via a random pre-sample, following
              the same pattern as run_imap_ga.py.
NSGA-II   : Standard constrained multi-objective EA (Deb et al. 2002).
C-TAEA    : Constrained Two-Archive EA — state-of-the-art specifically
              designed and evaluated on DAS-CMOP (Li et al. 2019).
Single-f1 : scipy differential evolution minimising only f1, with penalty
              for constraint violations.
Single-f2 : scipy differential evolution minimising only f2 (same).
Single-f3 : scipy differential evolution minimising only f3
              (tri-objective problems only).

Final comparison
----------------
The best feasible representative from each algorithm is pooled and ranked
via IMAP affine aggregation with equal objective weights — identical in
spirit to _run_imap_comparison in benchmark.py.
"""

from __future__ import annotations

import re
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

from pymoo.algorithms.moo.ctaea import CTAEA
from pymoo.algorithms.moo.moead import MOEAD, default_decomp
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from scipy.spatial.distance import cdist as _cdist
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.problems.multi.dascmop import (
    DASCMOP1, DASCMOP2, DASCMOP3,
    DASCMOP4, DASCMOP5, DASCMOP6,
    DASCMOP7, DASCMOP8, DASCMOP9,
)
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from allodyn.genetic_algorithms.imap_ga import IMAPGA
from allodyn.utils.imap_helpers import build_imap_config, compute_imap_scores, imap_best_from_front, make_pref_cv, run_imap_comparison
from allodyn.utils.imap_plotting import plot_preference_functions, savefig as _savefig_helper, bar_metric as _bar_metric_helper, heatmap_imap_winner as _heatmap_imap_winner_helper


# ── Output directories ─────────────────────────────────────────────────────────
_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SEED = 41

RESULTS_DIR = Path("tests/benchmark_dascmop_results") / f'{SEED}'
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
# Combined results across all problems (written at the very end)
CSV_PATH    = RESULTS_DIR / f"dascmop_combined_results.csv"


# ── Global configuration ───────────────────────────────────────────────────────
N_RUNS = 30   # independent runs per (problem, difficulty); seeds = SEED, SEED+1, …

# GA parameters shared by all evolutionary algorithms
N_GEN = 1000
POP_SIZE = 40

# IMAPGA settings
ARCHIVE = True

# Preference-function shape used throughout (IMAP-GA warmup, Pareto-front
# representative selection, and post-hoc IMAP comparison).
#   "linear"  : f(t) = t × 100            — proportional (default)
#   "convex"  : f(t) = t² × 100           — rewards proximity to optimum
#   "concave" : f(t) = √t × 100           — diminishing returns
#   "sigmoid" : logistic centred at t=0.5 — threshold / S-curve behaviour
# t = (hi - x) / (hi - lo) ∈ [0, 1]; higher t means lower (better) objective.
PREF_FN_SHAPE: str = "linear"

# Stakeholder configuration for IMAP-GA.
# Keyed by n_obj so bi-objective and tri-objective problems can have
# different stakeholder setups.
#
# Each value is a list of stakeholder dicts:
#   "obj_weights" : list[float]  — weight per objective, must sum to 1.0
#                                  and have length == n_obj.
#   "weight"      : float        — importance of this stakeholder relative
#                                  to others (normalised internally by IMAPGA).
#
# If n_obj is not present as a key, or if any obj_weights length mismatches
# n_obj, a single stakeholder with equal objective weights is used as fallback.
STAKEHOLDERS: dict[int, list[dict]] = {
    # Bi-objective problems (DAS-CMOP 1–6)
    2: [
        {"obj_weights": [0.5, 0.5], "weight": 0.3, "pref_shape": "linear"},
        {"obj_weights": [0.5, 0.5], "weight": 0.7, "pref_shape": "linear", "reverse": True}
    ],
    # Tri-objective problems (DAS-CMOP 7–9)
    3: [
        {"obj_weights": [1/3, 1/3, 1/3], "weight": 0.6, "pref_shape": "linear"},
        {"obj_weights": [0.2, 0.3, 0.5], "weight": 0.4, "pref_shape": "linear", "reverse": True}
    ],
}

# DAS-CMOP problem definitions — generated dynamically.
#   - DAS-CMOP1–6  : bi-objective, difficulty swept 1–16 (scalar int)
#   - DAS-CMOP7–9  : tri-objective, single entry with difficulty_factors=[0.5,0.5,0.5]
_BI_CLASSES  = [DASCMOP1, DASCMOP2, DASCMOP3, DASCMOP4, DASCMOP5, DASCMOP6]
_TRI_CLASSES = [DASCMOP7, DASCMOP8, DASCMOP9]
_BI_DIFFICULTIES = list(range(1, 17))   # 1 … 16

# 16 difficulty-factor triplets for DAS-CMOP7–9 (index = difficulty level 1–16)
_TRI_DIFFICULTIES: list[tuple[float, float, float]] = [
    (0.25, 0.00, 0.00),  # 1
    (0.00, 0.25, 0.00),  # 2
    (0.00, 0.00, 0.25),  # 3
    (0.25, 0.25, 0.25),  # 4
    (0.50, 0.00, 0.00),  # 5
    (0.00, 0.50, 0.00),  # 6
    (0.00, 0.00, 0.50),  # 7
    (0.50, 0.50, 0.50),  # 8
    (0.75, 0.00, 0.00),  # 9
    (0.00, 0.75, 0.00),  # 10
    (0.00, 0.00, 0.75),  # 11
    (0.75, 0.75, 0.75),  # 12
    (0.00, 1.00, 0.00),  # 13
    (0.50, 1.00, 0.00),  # 14
    (0.00, 1.00, 0.50),  # 15
    (0.50, 1.00, 0.50),  # 16
]

DASCMOP_PROBLEMS: list[dict] = []
for _i, _cls in enumerate(_BI_CLASSES, 1):
    for _d in _BI_DIFFICULTIES:
        DASCMOP_PROBLEMS.append({
            "id":          _i,
            "name":        f"DAS-CMOP{_i}",
            "cls":         _cls,
            "n_obj":       2,
            "ctor_kwargs": {"difficulty": _d},
            "n_gen":       N_GEN,
            "difficulty":  _d,
        })
for _i, _cls in enumerate(_TRI_CLASSES, 7):
    for _d, _factors in enumerate(_TRI_DIFFICULTIES, 1):
        DASCMOP_PROBLEMS.append({
            "id":          _i,
            "name":        f"DAS-CMOP{_i}",
            "cls":         _cls,
            "n_obj":       3,
            "ctor_kwargs": {"difficulty_factors": list(_factors)},
            "n_gen":       N_GEN,
            "difficulty":  _d,
        })

# ── Plot styling ───────────────────────────────────────────────────────────────
_PALETTE = {
    "IMAP-GA":  "#4CAF50",
    # "IMAP-GA (truncation)":  "#8BC34A",
    "NSGA-II":               "#2196F3",
    "NSGA-III":              "#1565C0",
    "C-TAEA":                "#FF9800",
    "MOEA/D-CDP":            "#00BCD4",
    "Single-f1":             "#E91E63",
    "Single-f2":             "#9C27B0",
    "Single-f3":             "#795548",
}
_FIG_DPI  = 150
_FIG_SIZE_WIDE = (16, 5)
_FIG_SIZE_SQ   = (8, 6)

# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AlgorithmResult:
    """Intermediate result from one algorithm run on one DAS-CMOP problem."""
    algorithm: str
    problem_name: str
    n_obj: int = 2
    # Full feasible Pareto front shape (k, n_obj)
    pareto_F: np.ndarray | None = None
    # Single representative (used in combined IMAP comparison)
    best_F: np.ndarray | None = None
    best_found_gen: int | None = None   # generation where best_F was first identified (IMAP-GA only)
    feasibility_rate: float = float("nan")
    n_feasible: int = 0
    wall_time_s: float = float("nan")

@dataclass
class RunRecord:
    """One row in the final results DataFrame."""
    problem_id: int
    problem_name: str          # e.g. "DAS-CMOP1_3" (base name + difficulty)
    n_obj: int
    algorithm: str
    difficulty: int | None = None   # scalar 1–16 for DAS-CMOP1-6; None for 7-9
    run_seed: int = SEED
    feasibility_rate: float = float("nan")
    n_feasible: int = 0
    wall_time_s: float = float("nan")
    # IMAP post-hoc comparison
    imap_score: float = float("nan")
    imap_winner: bool = False
    best_found_gen: int | None = None   # generation where best_F was found (IMAP-GA only)
    best_f1: float = float("nan")
    best_f2: float = float("nan")
    best_f3: float = float("nan")   # NaN for bi-objective problems
    # Preference scores per stakeholder per objective for the best solution.
    # Stored as a flat dict {s{sid}.{obj}: float}; expanded to columns in the
    # DataFrame by _record_to_row().  NaN for algorithms with no feasible best.
    pref_scores: dict = field(default_factory=dict)

def get_das_cmop_bounds(problem) -> list[tuple[float, float]]:
    n = problem.n_var
    m = problem.n_obj
    k = n - (m - 1)  # number of distance-function terms

    name = problem.__class__.__name__.upper()

    # All shape functions alpha_i have range [0, 1],
    # except f2 of CMOP3/CMOP6 which peaks at ~1.184
    shape_min = 0.0

    if any(p in name for p in ["DASCMOP1", "DASCMOP2", "DASCMOP3",
                                "DASCMOP9"]):
        # g = sum((x_j - sin(0.5π x_1))²), both in [0,1] → max term = 1
        g_min, g_max = 0.0, float(k)

    elif any(p in name for p in ["DASCMOP4", "DASCMOP5", "DASCMOP6",
                                  "DASCMOP7", "DASCMOP8"]):
        # g = k + sum((u)² - cos(20π u)),  u = x_j - 0.5 ∈ [-0.5, 0.5]
        # min per term: at u=0  →  0 - 1 = -1  →  g_min = k - k = 0
        # max per term: at u=0.45  →  0.2025 + 1 = 1.2025
        g_min, g_max = 0.0, k * (1 + 1.2025)  # = 2.2025 * k

    else:
        raise ValueError(f"Unknown DAS-CMOP problem: {name}")

    # Special case: f2 of CMOP3/CMOP6 has shape_max ≈ 1.184
    shape_maxes = []
    for i in range(m):
        if i == 1 and any(p in name for p in ["DASCMOP3", "DASCMOP6"]):
            shape_maxes.append(1.5 - 1.0 / 10**0.5)  # ≈ 1.184
        else:
            shape_maxes.append(1.0)

    return [(shape_min + g_min, s_max + g_max)
            for s_max in shape_maxes]

def _build_stakeholder_config(
    n_obj: int,
    obj_bounds: list[tuple[float, float]],
    obj_names: list[str],
) -> tuple[
    dict[int, dict[str, Callable[[float], float]]],
    dict[int, dict[str, float]],
    list[float],
]:
    """Thin wrapper around :func:`imap_helpers.build_imap_config` using the
    global STAKEHOLDERS config keyed by *n_obj*."""
    configs = STAKEHOLDERS.get(n_obj)
    if not configs:
        logger.warning(
            f"  No STAKEHOLDERS entry for n_obj={n_obj}; "
            "using single equal-weight stakeholder."
        )
    return build_imap_config(
        n_obj, obj_bounds, obj_names,
        stakeholder_configs=configs or None,
        default_shape=PREF_FN_SHAPE,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Shared indicator helpers
# ══════════════════════════════════════════════════════════════════════════════

def _feasible_F(res, problem: Problem) -> np.ndarray | None:
    """Return feasible objective vectors from a pymoo minimise result."""
    if res is None or res.F is None:
        return None
    if res.G is not None and res.G.size > 0:
        cv = np.sum(np.maximum(0.0, res.G), axis=1)
        mask = cv <= 1e-9
        F_feas = res.F[mask]
        return F_feas if len(F_feas) > 0 else None
    return res.F   # no constraint output → treat all as feasible

def _get_pareto_front(problem: Problem) -> np.ndarray | None:
    try:
        pf = problem.pareto_front()
        return pf if pf is not None and len(pf) > 0 else None # type: ignore
    except Exception:
        return None

# compute_imap_scores and imap_best_from_front imported from imap_helpers
_compute_imap_scores   = compute_imap_scores
_imap_best_from_front  = imap_best_from_front

# ══════════════════════════════════════════════════════════════════════════════
# MOEA/D-CDP  (pymoo MOEAD + Constraint Dominance Principle replacement rule)
# ══════════════════════════════════════════════════════════════════════════════

class MOEADCDP(MOEAD):
    """
    MOEA/D with Constraint Dominance Principle (CDP) constraint handling.

    Subclasses pymoo's MOEAD to:
    1. Remove the no-constraints assertion in ``_setup``.
    2. Override ``_replace`` so infeasible neighbours can be displaced by any
       solution with strictly lower constraint violation, and feasible
       neighbours are only displaced by a feasible offspring that is also
       better in decomposed objective space (Deb CDP rule).
    """

    def _setup(self, problem, **kwargs):
        # Identical to MOEAD._setup but without the has_constraints assertion.
        if self.ref_dirs is None:
            from pymoo.util.reference_direction import default_ref_dirs
            self.ref_dirs = default_ref_dirs(problem.n_obj)
        self.pop_size = len(self.ref_dirs)
        self.neighbors = np.argsort(
            _cdist(self.ref_dirs, self.ref_dirs), axis=1, kind="quicksort"
        )[:, :self.n_neighbors]
        if self.decomposition is None:
            self.decomposition = default_decomp(problem)

    def _replace(self, k, off):
        pop = self.pop
        N   = self.neighbors[k]

        FV     = self.decomposition.do(
            pop[N].get("F"), weights=self.ref_dirs[N, :], ideal_point=self.ideal # type: ignore
        )
        off_FV = self.decomposition.do(
            off.F[None, :], weights=self.ref_dirs[N, :], ideal_point=self.ideal
        )

        if self.problem.has_constraints(): # type: ignore
            G_nbrs  = pop[N].get("G") # type: ignore           # (n_neighbors, n_constr)
            G_off   = np.atleast_2d(off.get("G"))              # (1, n_constr)
            cv_nbrs = np.sum(np.maximum(0.0, G_nbrs), axis=1)  # (n_neighbors,)
            cv_off  = float(np.sum(np.maximum(0.0, G_off)))

            I = []
            for idx in range(len(N)):
                cv_n = float(cv_nbrs[idx])
                if cv_off == 0.0 and cv_n == 0.0:
                    # Both feasible: offspring wins only if better in decomposed space
                    if off_FV[idx] < FV[idx]:
                        I.append(idx)
                elif cv_off < cv_n:
                    # Offspring has strictly lower violation → CDP wins regardless
                    I.append(idx)
                # else: neighbour is feasible (or less violated) → keep it
            pop[N[np.array(I, dtype=int)]] = off # type: ignore
        else:
            I = np.where(off_FV < FV)[0]
            pop[N[I]] = off # type: ignore

# ══════════════════════════════════════════════════════════════════════════════
# Algorithm runners
# ══════════════════════════════════════════════════════════════════════════════

def _run_nsga2(problem: Problem, n_obj: int, n_gen: int = N_GEN, seed: int = SEED) -> AlgorithmResult:
    result = AlgorithmResult(
        algorithm="NSGA-II", problem_name="", n_obj=n_obj
    )
    try:
        algorithm = NSGA2(pop_size=POP_SIZE)
        t0 = time.perf_counter()
        res = pymoo_minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
        result.wall_time_s = time.perf_counter() - t0

        F_feas = _feasible_F(res, problem)
        n_total = len(res.F) if res.F is not None else 0
        n_feas  = len(F_feas) if F_feas is not None else 0
        result.feasibility_rate = n_feas / n_total * 100 if n_total > 0 else 0.0
        result.n_feasible = n_feas
        result.pareto_F   = F_feas

        logger.info(
            f"  NSGA-II : feasible={n_feas}/{n_total}, t={result.wall_time_s:.1f}s"
        )
    except Exception:
        logger.error(f"  NSGA-II failed:\n{traceback.format_exc()}")
    return result

def _run_nsga3(problem: Problem, n_obj: int, n_gen: int = N_GEN, seed: int = SEED) -> AlgorithmResult:
    result = AlgorithmResult(
        algorithm="NSGA-III", problem_name="", n_obj=n_obj
    )
    try:
        n_partitions = 12   # → 91 reference directions for 3 objectives
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        algorithm = NSGA3(ref_dirs=ref_dirs)
        t0 = time.perf_counter()
        res = pymoo_minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
        result.wall_time_s = time.perf_counter() - t0

        F_feas = _feasible_F(res, problem)
        n_total = len(res.F) if res.F is not None else 0
        n_feas  = len(F_feas) if F_feas is not None else 0
        result.feasibility_rate = n_feas / n_total * 100 if n_total > 0 else 0.0
        result.n_feasible = n_feas
        result.pareto_F   = F_feas

        logger.info(
            f"  NSGA-III: feasible={n_feas}/{n_total}, t={result.wall_time_s:.1f}s"
        )
    except Exception:
        logger.error(f"  NSGA-III failed:\n{traceback.format_exc()}")
    return result

def _run_ctaea(problem: Problem, n_obj: int, n_gen: int = N_GEN, seed: int = SEED) -> AlgorithmResult:
    result = AlgorithmResult(
        algorithm="C-TAEA", problem_name="", n_obj=n_obj
    )
    try:
        n_partitions = 99 if n_obj == 2 else 12   # ~100 or ~91 directions
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        algorithm = CTAEA(ref_dirs=ref_dirs)
        t0 = time.perf_counter()
        res = pymoo_minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
        result.wall_time_s = time.perf_counter() - t0

        F_feas = _feasible_F(res, problem)
        n_total = len(res.F) if res.F is not None else 0
        n_feas  = len(F_feas) if F_feas is not None else 0
        result.feasibility_rate = n_feas / n_total * 100 if n_total > 0 else 0.0
        result.n_feasible = n_feas
        result.pareto_F   = F_feas

        logger.info(
            f"  C-TAEA  : feasible={n_feas}/{n_total}, t={result.wall_time_s:.1f}s"
        )
    except Exception:
        logger.error(f"  C-TAEA failed:\n{traceback.format_exc()}")
    return result

def _run_moeadcdp(problem: Problem, n_obj: int, n_gen: int = N_GEN, seed: int = SEED) -> AlgorithmResult:
    result = AlgorithmResult(
        algorithm="MOEA/D-CDP", problem_name="", n_obj=n_obj
    )
    try:
        n_partitions = 99 if n_obj == 2 else 12   # ~100 or ~91 directions
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
        algorithm = MOEADCDP(ref_dirs=ref_dirs)
        t0 = time.perf_counter()
        res = pymoo_minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_gen),
            seed=seed,
            verbose=False,
        )
        result.wall_time_s = time.perf_counter() - t0

        F_feas = _feasible_F(res, problem)
        n_total = len(res.F) if res.F is not None else 0
        n_feas  = len(F_feas) if F_feas is not None else 0
        result.feasibility_rate = n_feas / n_total * 100 if n_total > 0 else 0.0
        result.n_feasible = n_feas
        result.pareto_F   = F_feas

        logger.info(
            f"  MOEA/D-CDP: feasible={n_feas}/{n_total}, t={result.wall_time_s:.1f}s"
        )
    except Exception:
        logger.error(f"  MOEA/D-CDP failed:\n{traceback.format_exc()}")
    return result

def _run_imap_ga(
    problem: Problem,
    n_obj: int,
    pref_fns: dict[int, dict[str, Callable[[float], float]]],
    obj_weights: dict[int, dict[str, float]],
    s_weights: list[float],
    obj_names: list[str],
    n_gen: int = N_GEN,
    ts: bool = True,
    cv_pref_fn: Callable[[float], float] | None = None,
    seed: int = SEED,
) -> AlgorithmResult:
    """
    Run IMAP-GA on a DAS-CMOP problem.

    The original constrained problem is passed directly (no penalty wrapper),
    consistent with run_imap_ga.py.
    """
    # label = f"IMAP-GA (tournament)" if ts else "IMAP-GA (truncation)"
    label = "IMAP-GA"
    result = AlgorithmResult(
        algorithm=label, problem_name="", n_obj=n_obj
    )
    try:
        algorithm = IMAPGA(
            preference_functions=pref_fns,
            objective_weights=obj_weights,
            stakeholder_weights=s_weights,
            all_objectives=obj_names,
            pop_size=POP_SIZE,
            tournament_selection=ts,
            archive=ARCHIVE,
            cv_pref_fn=cv_pref_fn,
            cv_weight=0.5,
            seed=seed,
            verbose=False,
        )

        n_stakeholders = len(pref_fns)
        shapes = [
            STAKEHOLDERS.get(n_obj, [{}])[i].get("pref_shape", PREF_FN_SHAPE)
            for i in range(n_stakeholders)
        ]
        cv_info = f", cv_weight=0.5" if cv_pref_fn is not None else ""
        logger.info(
            f"  {label}: n_stakeholders={n_stakeholders}, "
            f"pref_shapes={shapes}, stakeholder_weights={s_weights}{cv_info}"
        )

        t0 = time.perf_counter()
        res = pymoo_minimize(
            problem,
            algorithm,
            get_termination("n_gen", n_gen),
            seed=seed,
            verbose=False,
            copy_algorithm=False,
        )
        result.wall_time_s = time.perf_counter() - t0

        # ── Re-evaluate final population on original problem ──────────────────
        final_pop = res.pop
        if final_pop is not None and len(final_pop) > 0:
            X_final = final_pop.get("X")
            pop_out = problem.evaluate(X_final, return_as_dictionary=True)
            F_all = pop_out["F"] # type: ignore

            if problem.n_constr > 0 and "G" in pop_out and pop_out["G"] is not None: # type: ignore
                cv_all   = np.sum(np.maximum(0.0, pop_out["G"]), axis=1) # type: ignore
                feas_mask = cv_all <= 1e-9
            else:
                feas_mask = np.ones(len(F_all), dtype=bool)

            n_total = len(F_all)
            n_feas  = int(feas_mask.sum())
            result.feasibility_rate = n_feas / n_total * 100 if n_total > 0 else 0.0
            result.n_feasible = n_feas

        # ── Read current_best objective values ────────────────────────────────
        # Only feasible solutions are stored; an infeasible current_best is
        # excluded so it cannot enter the post-hoc IMAP comparison.
        cb = res.algorithm.current_best # type: ignore
        result.best_found_gen = res.algorithm.current_best_gen # type: ignore
        if cb is not None:
            F_cb = cb.get("F")
            F_true = F_cb[0] if F_cb.ndim == 2 else F_cb  # (1, n_obj) → (n_obj,)

            is_feasible = True
            if problem.n_constr > 0:
                G_cb = cb.get("G")
                if G_cb is not None:
                    cv = float(np.sum(np.maximum(0.0, G_cb.flatten())))
                    is_feasible = cv <= 1e-9

            if is_feasible:
                result.best_F = F_true
                result.pareto_F = np.array([F_true])
            else:
                logger.warning(f"  {label}: current_best is infeasible — excluded from IMAP comparison")

        logger.info(
            f"  {label}: feasible={result.n_feasible}, best_gen={result.best_found_gen}, "
            f"t={result.wall_time_s:.1f}s"
        )
    except Exception:
        logger.error(f"  {label} failed:\n{traceback.format_exc()}")
    return result

# ══════════════════════════════════════════════════════════════════════════════
# IMAP post-hoc comparison  (mirrors _run_imap_comparison in benchmark.py)
# ══════════════════════════════════════════════════════════════════════════════

def _run_imap_comparison(
    algo_results: list[AlgorithmResult],
    obj_names: list[str],
    problem_name: str,
    pref_fns: dict[int, dict[str, Callable[[float], float]]],
    obj_weights: dict[int, dict[str, float]],
    s_weights: list[float],
) -> dict[str, float]:
    """Thin wrapper around :func:`imap_helpers.run_imap_comparison`.

    Also appends best_found_gen to log lines for IMAP-GA entries.
    """
    gen_map = {ar.algorithm: ar.best_found_gen for ar in algo_results}
    labels   = [ar.algorithm for ar in algo_results]
    best_Fs  = [ar.best_F for ar in algo_results]

    score_map = run_imap_comparison(
        labels, best_Fs, obj_names, pref_fns, obj_weights, s_weights, # type: ignore
        context_label=problem_name, logger=logger,
    )

    # Append generation info for IMAP-GA entries
    for lbl, gen in gen_map.items():
        if gen is not None and lbl in score_map:
            logger.info(f"    {lbl:<14s}: best_gen={gen}")

    return score_map

# ══════════════════════════════════════════════════════════════════════════════
# Single-problem orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def _run_problem(
    prob_entry: dict,
    *,
    seed: int = SEED,
    plot_pref_fns: bool = False,
    prob_plots_dir: Path | None = None,
) -> tuple[list[RunRecord], dict]:
    """
    Run all algorithms on one DAS-CMOP problem entry (one difficulty level).

    Parameters
    ----------
    prob_entry:
        Entry from DASCMOP_PROBLEMS.
    plot_pref_fns:
        When True, save a preference-function plot to *prob_plots_dir*.
        Pass True only for the first difficulty of each problem to avoid
        saving the same plot 16 times (bounds don't change with difficulty).
    prob_plots_dir:
        Directory to write the preference-function plot to.

    Returns (list[RunRecord], algo_result_store) where the store maps
    algorithm label → AlgorithmResult (used by scatter plots).
    """
    prob_id:    int       = prob_entry["id"]
    prob_name:  str       = prob_entry["name"]
    n_obj:      int       = prob_entry["n_obj"]
    n_gen:      int       = prob_entry.get("n_gen", N_GEN)
    difficulty: int | None = prob_entry.get("difficulty")
    records:    list[RunRecord] = []
    ar_store:   dict = {}

    if difficulty is not None:
        factors = prob_entry["ctor_kwargs"].get("difficulty_factors")
        if factors is not None:
            d_label = f"  difficulty={difficulty}  factors={factors}"
        else:
            d_label = f"  difficulty={difficulty}"
    else:
        d_label = ""
    logger.info(f"{'─'*80}")
    logger.info(f"Problem: {prob_name}{d_label}")

    # Instantiate problem
    try:
        problem: Problem = prob_entry["cls"](**prob_entry["ctor_kwargs"])
    except Exception:
        logger.error(f"  Failed to instantiate {prob_name}:\n{traceback.format_exc()}")
        return records, ar_store

    logger.info(
        f"  n_var={problem.n_var}, n_obj={problem.n_obj}, "
        f"n_constr={problem.n_constr}, n_gen={n_gen}"
    )

    # Objective names
    obj_names = [f"f{j+1}" for j in range(n_obj)]

    # ── Estimate objective bounds for IMAP preference functions ───────────────
    logger.info("  Estimating objective bounds via random sample ...")
    obj_bounds = get_das_cmop_bounds(problem)
    for j, (lo, hi) in enumerate(obj_bounds):
        logger.info(f"  f{j+1} ∈ [{lo:.4f}, {hi:.4f}]")
    
    # Get the preference functions
    pref_fns, obj_weights, s_weights = _build_stakeholder_config(
        n_obj, obj_bounds, obj_names
    )

    if plot_pref_fns and prob_plots_dir is not None:
        plot_preference_functions(
            pref_fns, obj_bounds, obj_names,
            save_path=prob_plots_dir / "pref_fns.png",
        )

    # ── Run all algorithms ────────────────────────────────────────────────────
    algo_results: list[AlgorithmResult] = []

    # Use CV as a 50 % aggregation signal for constrained problems so that
    # feasible individuals always dominate infeasible ones via the preference
    # score (f(0)=100, f(ε)=10, f→0 as cv→∞).
    _cv_pref_fn = make_pref_cv() if problem.n_constr > 0 else None

    logger.info("  Running IMAP-GA ...")
    r = _run_imap_ga(problem, n_obj, pref_fns, obj_weights, s_weights, obj_names, n_gen=n_gen, ts=True, cv_pref_fn=_cv_pref_fn, seed=seed)
    r.problem_name = prob_name
    algo_results.append(r)

    # logger.info("  Running IMAP-GA (truncated) ...")
    # r = _run_imap_ga(problem, n_obj, pref_fns, obj_weights, s_weights, obj_names, n_gen=n_gen, ts=False, cv_pref_fn=_cv_pref_fn)
    # r.problem_name = prob_name
    # algo_results.append(r)

    if n_obj == 2:
        logger.info("  Running NSGA-II ...")
        r = _run_nsga2(problem, n_obj, n_gen=n_gen, seed=seed)
    else:
        logger.info("  Running NSGA-III ...")
        r = _run_nsga3(problem, n_obj, n_gen=n_gen, seed=seed)
    r.problem_name = prob_name
    algo_results.append(r)

    logger.info("  Running C-TAEA ...")
    r = _run_ctaea(problem, n_obj, n_gen=n_gen, seed=seed)
    r.problem_name = prob_name
    algo_results.append(r)

    logger.info("  Running MOEA/D-CDP ...")
    r = _run_moeadcdp(problem, n_obj, n_gen=n_gen, seed=seed)
    r.problem_name = prob_name
    algo_results.append(r)

    # ── Pick representative best_F from Pareto-front algorithms ──────────────
    # NSGA-II and C-TAEA return a front; use IMAP to elect one representative.
    for ar in algo_results:
        if ar.best_F is None and ar.pareto_F is not None and len(ar.pareto_F) > 0:
            ar.best_F = _imap_best_from_front(ar.pareto_F, obj_names, pref_fns, obj_weights, s_weights)

    # ── IMAP post-hoc comparison ──────────────────────────────────────────────
    logger.info("  Running IMAP post-hoc comparison ...")
    score_map = _run_imap_comparison(algo_results, obj_names, prob_name, pref_fns, obj_weights, s_weights)

    finite_scores = [v for v in score_map.values() if np.isfinite(v)]
    max_score = max(finite_scores) if finite_scores else float("nan")

    # ── Build RunRecords and store results ────────────────────────────────────
    run_name = f"{prob_name}_{difficulty}" if difficulty is not None else prob_name

    for ar in algo_results:
        ar_store[ar.algorithm] = ar
        imap_score = score_map.get(ar.algorithm, float("nan"))
        imap_winner = (
            np.isfinite(imap_score)
            and np.isfinite(max_score)
            and abs(imap_score - max_score) < 1e-6
        )

        best_f1 = float(ar.best_F[0]) if ar.best_F is not None else float("nan")
        best_f2 = float(ar.best_F[1]) if ar.best_F is not None else float("nan")
        best_f3 = (
            float(ar.best_F[2])
            if ar.best_F is not None and len(ar.best_F) > 2
            else float("nan")
        )

        # Preference scores per stakeholder per objective for this algorithm's
        # best solution.  NaN for all entries when no feasible solution was found.
        pref_scores: dict[str, float] = {}
        for sid in sorted(pref_fns.keys()):
            for j, obj_name in enumerate(obj_names):
                col = f"s{sid}.{obj_name}"
                if ar.best_F is not None:
                    pref_scores[col] = float(pref_fns[sid][obj_name](float(ar.best_F[j])))
                else:
                    pref_scores[col] = float("nan")

        records.append(RunRecord(
            problem_id=prob_id,
            problem_name=run_name,
            n_obj=n_obj,
            algorithm=ar.algorithm,
            difficulty=difficulty,
            run_seed=seed,
            feasibility_rate=ar.feasibility_rate,
            n_feasible=ar.n_feasible,
            wall_time_s=ar.wall_time_s,
            imap_score=imap_score,
            imap_winner=imap_winner,
            best_found_gen=ar.best_found_gen,
            best_f1=best_f1,
            best_f2=best_f2,
            best_f3=best_f3,
            pref_scores=pref_scores,
        ))

    return records, ar_store

# ══════════════════════════════════════════════════════════════════════════════
# Visualizations  (follows the style of benchmark.py plot functions)
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, name: str) -> None:  # type: ignore
    _savefig_helper(fig, PLOTS_DIR / name, dpi=_FIG_DPI, logger=logger)

def _bar_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    hline: float | None = None,
) -> None:
    """Grouped bar chart over all DAS-CMOP problems."""
    fig = _bar_metric_helper(
        df, metric, ylabel, title,
        group_col="problem_name",
        palette=_PALETTE,
        hline=hline,
        fig_size=_FIG_SIZE_WIDE,
        dpi=_FIG_DPI,
        logger=logger,
    )
    _savefig(fig, filename)

def _heatmap_imap_winner(df: pd.DataFrame) -> None:
    """Heatmap: which algorithm won the IMAP post-hoc comparison per problem."""
    _heatmap_imap_winner_helper(
        df,
        save_path=PLOTS_DIR / f"06_imap_winner_heatmap.png",
        row_col="problem_name",
        palette=_PALETTE,
        title="IMAP Post-hoc Comparison Winner per Problem  (1 = winner)",
        fig_size=(11, 5),
        dpi=_FIG_DPI,
        logger=logger,
    )

def _scatter_pareto_fronts(
    df: pd.DataFrame,
    algo_results_all: dict[tuple[str, int | None, str], AlgorithmResult],
    *,
    rep_difficulty: int = 8,
    save_dir: Path | None = None,
    filename: str | None = None,
) -> None:
    """
    Scatter plot of Pareto front / best solution for each algorithm.

    Shows the first 3 unique bi-objective problem names at *rep_difficulty*
    (or the closest available difficulty if that exact one is absent).

    Parameters
    ----------
    rep_difficulty:
        Which difficulty level to use as the representative scatter.
    save_dir:
        Directory to write the plot to.  Falls back to ``PLOTS_DIR``.
    filename:
        Output file name.  Defaults to ``07_pareto_fronts.png``.
    """
    save_dir = save_dir or PLOTS_DIR
    filename = filename or f"07_pareto_fronts.png"

    # problem_name in df is now "DAS-CMOP1_3" — strip the suffix to get the base
    # name used as the key in algo_results_all.
    bi_base_names = sorted(
        df[df["n_obj"] == 2]["problem_name"]
        .str.rsplit("_", n=1).str[0]
        .unique()
    )[:3]
    if not bi_base_names:
        return

    fig, axes = plt.subplots(1, len(bi_base_names), figsize=(5 * len(bi_base_names), 5))
    if len(bi_base_names) == 1:
        axes = [axes]

    for ax, prob_name in zip(axes, bi_base_names):
        # Find the closest available difficulty to rep_difficulty
        avail_diffs = sorted({
            d for (pn, d, _) in algo_results_all if pn == prob_name and d is not None
        })
        if not avail_diffs:
            continue
        chosen_d = min(avail_diffs, key=lambda x: abs(x - rep_difficulty))

        for algo, color in _PALETTE.items():
            ar = algo_results_all.get((prob_name, chosen_d, algo))
            if ar is None:
                continue
            if ar.pareto_F is not None and len(ar.pareto_F) > 0:
                ax.scatter(
                    ar.pareto_F[:, 0], ar.pareto_F[:, 1],
                    s=12, alpha=0.65, color=color, label=algo, edgecolors="none",
                )
            if ar.best_F is not None:
                ax.scatter(
                    ar.best_F[0], ar.best_F[1],
                    s=90, color=color, marker="*",
                    edgecolors="black", linewidths=0.5,
                )
        ax.set_xlabel("f1")
        ax.set_ylabel("f2")
        ax.set_title(f"{prob_name}  (d={chosen_d})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, markerscale=1.3)

    fig.suptitle(
        f"Pareto Fronts by Algorithm  (★ = IMAP-selected representative, difficulty={chosen_d})",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _savefig_helper(fig, save_dir / filename, dpi=_FIG_DPI, logger=logger)


def _bar_win_count(df: pd.DataFrame) -> None:
    """Bar chart of total IMAP wins per algorithm across all (problem, difficulty) runs."""
    total_runs = df["problem_name"].nunique()
    win_counts = (
        df[df["imap_winner"]]
        .groupby("algorithm")
        .size()
        .reindex(list(_PALETTE.keys()), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=_FIG_SIZE_SQ)
    colors = [_PALETTE.get(a, "#888888") for a in win_counts.index]
    ax.bar(win_counts.index, win_counts.values, color=colors, edgecolor="white") # type: ignore
    ax.set_xlabel("Algorithm")
    ax.set_ylabel(f"Number of IMAP wins (out of {total_runs})")
    ax.set_ylim(0, total_runs + 0.5)
    ax.set_title("IMAP Win Count Across All DAS-CMOP Problems and Difficulties",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    _savefig(fig, f"08_imap_win_count.png")

def generate_problem_plots(
    df_prob: pd.DataFrame,
    prob_name: str,
    prob_plots_dir: Path,
    algo_results_all: dict[tuple[str, int | None, str], AlgorithmResult],
) -> None:
    """
    Generate per-problem plots with difficulty on the x-axis.

    Called immediately after all difficulties for one problem have been run,
    so progress is persisted incrementally.
    """
    sns.set_theme(style="whitegrid", font_scale=1.0)

    def _save(fig: plt.Figure, name: str) -> None: # type: ignore
        _savefig_helper(fig, prob_plots_dir / name, dpi=_FIG_DPI, logger=logger)

    def _bar(metric: str, ylabel: str, title: str, fname: str,
             hline: float | None = None) -> None:
        has_difficulty = df_prob["difficulty"].notna().any()
        group_col = "difficulty" if has_difficulty else "problem_name"
        fig = _bar_metric_helper(
            df_prob, metric, ylabel, title,
            group_col=group_col,
            palette=_PALETTE,
            hline=hline,
            fig_size=(max(10, len(df_prob["difficulty"].unique()) * 0.8), 5),
            dpi=_FIG_DPI,
            logger=logger,
        )
        _save(fig, fname)

    ts = _TIMESTAMP
    _bar("imap_score",      "IMAP Score (higher is better)",
         f"{prob_name}: IMAP Score by Difficulty",
         f"{ts}_01_imap_score.png")
    _bar("feasibility_rate","Feasibility Rate (%)",
         f"{prob_name}: Feasibility Rate by Difficulty",
         f"{ts}_02_feasibility.png", hline=100.0)
    _bar("wall_time_s",     "Wall Time (s)",
         f"{prob_name}: Runtime by Difficulty",
         f"{ts}_03_wall_time.png")

    # IMAP winner heatmap (difficulty × algorithm)
    _heatmap_imap_winner_helper(
        df_prob,
        save_path=prob_plots_dir / f"{ts}_04_imap_winner_heatmap.png",
        row_col="difficulty",
        col_col="algorithm",
        winner_col="imap_winner",
        palette=_PALETTE,
        title=f"{prob_name}: IMAP Winner by Difficulty",
        fig_size=(11, max(4, len(df_prob["difficulty"].unique()) * 0.4 + 1)),
        dpi=_FIG_DPI,
        logger=logger,
    )

    # Scatter: representative difficulty (mid-range d=8) if bi-objective
    if df_prob["n_obj"].iloc[0] == 2:
        _scatter_pareto_fronts(
            df_prob, algo_results_all,
            rep_difficulty=8,
            save_dir=prob_plots_dir,
            filename=f"{ts}_05_pareto_fronts_d8.png",
        )

    logger.success(f"  {prob_name} plots saved → {prob_plots_dir}")

def generate_experiment_plots(
    df: pd.DataFrame,
    algo_results_all: dict[tuple[str, int | None, str], AlgorithmResult],
) -> None:
    """
    Generate global summary plots aggregated across all difficulties.

    Metrics are averaged over difficulties so each (problem, algorithm) pair
    becomes one bar in the charts.
    """
    logger.info("Generating global summary visualizations ...")
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # problem_name in df is "DAS-CMOP1_3"; strip the suffix to get the base
    # name for aggregating across all difficulty levels of the same problem.
    df_agg = (
        df.assign(problem_name=df["problem_name"].str.rsplit("_", n=1).str[0])
        .groupby(["problem_name", "algorithm"], sort=False)
        .agg(
            n_obj=("n_obj", "first"),
            imap_score=("imap_score", "mean"),
            feasibility_rate=("feasibility_rate", "mean"),
            wall_time_s=("wall_time_s", "mean"),
            imap_winner=("imap_winner", "sum"),   # count wins, not mean
        )
        .reset_index()
    )
    # Re-label imap_winner as a bool: True if this algo won in any difficulty
    df_agg["imap_winner"] = df_agg["imap_winner"] > 0

    _bar_metric(
        df_agg, "imap_score", "Mean IMAP Score (higher is better)",
        "Mean IMAP Score by Problem and Algorithm (averaged over difficulties)",
        f"01_imap_score.png",
    )
    _bar_metric(
        df_agg, "feasibility_rate", "Mean Feasibility Rate (%)",
        "Mean Feasibility Rate by Problem and Algorithm (averaged over difficulties)",
        f"02_feasibility.png",
        hline=100.0,
    )
    _bar_metric(
        df_agg, "wall_time_s", "Mean Wall Time (s)",
        "Mean Algorithm Runtime by Problem (averaged over difficulties)",
        f"03_wall_time.png",
    )
    _heatmap_imap_winner(df_agg)
    _scatter_pareto_fronts(df, algo_results_all)
    _bar_win_count(df)

    logger.success(f"Global summary plots saved to: {PLOTS_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def _record_to_row(r: RunRecord) -> dict:
    """Flatten a RunRecord to a plain dict, expanding pref_scores inline."""
    d = {k: v for k, v in r.__dict__.items() if k != "pref_scores"}
    d.update(r.pref_scores)
    return d

def _sanitise_npz_key(label: str) -> str:
    """Make an algorithm label safe as a numpy .npz array key."""
    return re.sub(r"[^A-Za-z0-9_]", "_", label).strip("_")

# Labels of IMAP-GA variants — only the representative point is stored for
# these (not the full front) because their pareto_F is already a single point.
# _IMAP_LABELS = {"IMAP-GA (tournament)", "IMAP-GA (truncation)"}

def _save_pareto_npz(
    ar_store: dict[str, "AlgorithmResult"],
    obj_names: list[str],
    prob_dir: Path,
    difficulty: int | None,
) -> None:
    """
    Save Pareto-front data to a compact compressed NumPy archive for later
    3-D (or 2-D) plotting without re-running the benchmark.

    Layout of the .npz file
    -----------------------
    ``obj_names``               — string array of objective names, e.g. ``["f1","f2","f3"]``
    ``<algo_key>``              — float32 array of shape ``(k, n_obj)``:
                                  * baseline algorithms : full feasible Pareto front (k ≥ 1)
                                  * IMAP-GA variants    : single representative row (k = 1)

    Index arrays (for the visualiser)
    ----------------------------------
    ``_keys``    — sanitised array keys, one per saved algorithm
    ``_labels``  — original human-readable algorithm labels
    ``_types``   — ``"front"`` (full Pareto front) or ``"best"`` (single point)

    How to load  (use pareto_viz.ipynb instead of doing this manually)
    -----------
    ::

        data      = np.load("path/to/file.npz", allow_pickle=False)
        obj_names = list(data["obj_names"])
        labels    = list(data["_labels"])
        keys      = list(data["_keys"])
        types     = list(data["_types"])
        for key, label, typ in zip(keys, labels, types):
            F = data[key]   # shape (k, n_obj)
    """
    arrays: dict[str, np.ndarray] = {"obj_names": np.array(obj_names)}

    saved_keys:   list[str] = []
    saved_labels: list[str] = []
    saved_types:  list[str] = []   # "front" | "best"

    for algo_label, ar in ar_store.items():
        key = _sanitise_npz_key(algo_label)
        # if algo_label in _IMAP_LABELS:
        # Store only the IMAP-selected representative (single point)
        if ar.best_F is not None:
            arrays[key] = ar.best_F.reshape(1, -1).astype(np.float32)
            saved_keys.append(key)
            saved_labels.append(algo_label)
            saved_types.append("best")
        else:
            # Store the full feasible Pareto front
            if ar.pareto_F is not None and len(ar.pareto_F) > 0:
                arrays[key] = ar.pareto_F.astype(np.float32)
                saved_keys.append(key)
                saved_labels.append(algo_label)
                saved_types.append("front")
            # Also store the IMAP-selected best representative as a star marker
            if ar.best_F is not None:
                best_key = f"{key}_best"
                arrays[best_key] = ar.best_F.reshape(1, -1).astype(np.float32)
                saved_keys.append(best_key)
                saved_labels.append(algo_label)
                saved_types.append("best")

    # Index arrays — used by the visualiser to reconstruct human-readable
    # labels without reverse-engineering the key sanitisation.
    arrays["_keys"]   = np.array(saved_keys)
    arrays["_labels"] = np.array(saved_labels)
    arrays["_types"]  = np.array(saved_types)

    suffix = f"_d{difficulty}" if difficulty is not None else ""
    out_path = prob_dir / f"pareto{suffix}.npz"
    np.savez_compressed(out_path, **arrays) # type: ignore
    logger.info(f"  Pareto fronts saved → {out_path}")

def run_all_experiments() -> tuple[pd.DataFrame, dict[tuple[str, int | None, str], AlgorithmResult]]:
    all_records:    list[RunRecord] = []
    algo_results_all: dict[tuple[str, int | None, str], AlgorithmResult] = {}

    # Group problem entries by problem ID so we can save results per problem
    problems_by_id: dict[int, list[dict]] = {}
    for pe in DASCMOP_PROBLEMS:
        problems_by_id.setdefault(pe["id"], []).append(pe)

    for prob_id, prob_entries in problems_by_id.items():
        if not (prob_id == 8 or prob_id == 4):
            continue
        prob_name = prob_entries[0]["name"]
        logger.info(f"{'█'*80}")
        logger.info(f"  PROBLEM GROUP {prob_id}: {prob_name}  "
                    f"({len(prob_entries)} difficulty level(s))")
        logger.info(f"{'█'*80}")

        # Per-problem output directories
        prob_dir       = RESULTS_DIR / prob_name
        prob_plots_dir = prob_dir / "plots"
        prob_dir.mkdir(parents=True, exist_ok=True)
        prob_plots_dir.mkdir(parents=True, exist_ok=True)

        prob_records: list[RunRecord] = []

        for entry_idx, prob_entry in enumerate(prob_entries):
            entry_difficulty = prob_entry.get("difficulty")
            if entry_difficulty != 1:
                continue

            for run_idx in range(N_RUNS):
                seed = SEED + run_idx
                logger.info(f"  ── Run {run_idx + 1}/{N_RUNS} (seed={seed}) ──")
                records, ar_store = _run_problem(
                    prob_entry,
                    seed=seed,
                    plot_pref_fns=(entry_idx == 0 and run_idx == 0),
                    prob_plots_dir=prob_plots_dir,
                )

                prob_records.extend(records)
                all_records.extend(records)
                for algo_label, ar in ar_store.items():
                    algo_results_all[(prob_name, entry_difficulty, algo_label)] = ar

                obj_names = [f"f{j+1}" for j in range(prob_entry["n_obj"])]
                _save_pareto_npz(ar_store, obj_names, prob_dir, entry_difficulty)

        # ── Save per-problem results ──────────────────────────────────────────
        if prob_records:
            df_prob = pd.DataFrame([_record_to_row(r) for r in prob_records])
            prob_csv = prob_dir / f"results.csv"
            df_prob.to_csv(prob_csv, index=False)
            logger.success(f"  {prob_name} results saved → {prob_csv}")

            # ── Per-problem plots ─────────────────────────────────────────────
            try:
                generate_problem_plots(df_prob, prob_name, prob_plots_dir, algo_results_all)
            except Exception:
                logger.error(
                    f"  {prob_name} per-problem plots failed:\n{traceback.format_exc()}"
                )

    # ── Combined results across all problems ─────────────────────────────────
    df = pd.DataFrame([_record_to_row(r) for r in all_records])
    df.to_csv(CSV_PATH, index=False)
    logger.success(f"Combined results saved → {CSV_PATH}")
    return df, algo_results_all

if __name__ == "__main__":
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
        colorize=True,
    )
    logger.add(
        RESULTS_DIR / f"benchmark_dascmop.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG",
        rotation="10 MB",
    )

    logger.info("Starting DAS-CMOP Benchmark Suite")
    df, algo_results_all = run_all_experiments()

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", "{:.4f}".format)

    summary_cols = [
        "problem_id", "problem_name", "difficulty", "run_seed", "n_obj", "algorithm",
        "feasibility_rate", "n_feasible",
        "imap_score", "imap_winner",
        "best_found_gen",
        "best_f1", "best_f2", "best_f3",
        "wall_time_s",
    ]
    print(df[[c for c in summary_cols if c in df.columns]].to_string(index=False))

    generate_experiment_plots(df, algo_results_all)
    logger.success("DAS-CMOP benchmark complete.")
