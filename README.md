# IGS-GA-II

Multi-objective constrained optimisation using a **Genetic Algorithm** with
**IMAP/Preferendus** preference-aggregation survival selection.

---

## What is this?

The **Preferendus** method evaluates design alternatives by converting raw objective
values (f1, f2, …) into *preference scores* via stakeholder-defined preference
functions, then aggregating those scores using a population-relative z-score
normalisation (`a_fine_aggregator`).  The key property: **the same solution can
receive a different score in a different generation**, because scoring is relative
to the current population — not absolute.

**IGS-GA** (IMAP Genetic Algorithm) replaces NSGA-II's non-dominated sorting and
crowding-distance survival with a two-pass IMAP affine aggregation.  Any
`pymoo`-compatible problem can be solved by:

1. defining preference functions and objective bounds for each objective
2. configuring stakeholder weights and preference shapes
3. constructing an `IGSGA` instance and passing it to `pymoo_minimize`

---

## Repository structure

```
IGS-GA-II/
├── src/
│   └── optimization/
│       ├── igs_ga.py          # IGSGA algorithm (IMAPSurvival + IGSGA class)
│       └── a_fine_aggregator.py  # affine aggregator kernel (no external deps)
├── examples/
│   └── das-cmop.ipynb         # demo notebook (DAS-CMOP benchmark suite)
├── benchmark_dascmop.py       # full benchmark: IGS-GA vs NSGA-II, C-TAEA, MOEA/D-CDP
└── pyproject.toml
```

---

## Installation

```bash
pip install -e .
```

Runtime dependencies: `numpy`, `pymoo`.

---

## Quick start

```python
import numpy as np
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.problems.multi.dascmop import DASCMOP1
from pymoo.termination import get_termination

from optimization.igs_ga import IGSGA

# 1. Define objective bounds
problem  = DASCMOP1(difficulty=4)
k        = problem.n_var - (problem.n_obj - 1)  # distance-function terms
f_lo, f_hi = 0.0, 1.0 + k                        # analytical bounds for DASCMOP1

# 2. Build preference functions (linear, minimisation)
def linear_pref(lo, hi):
    def _fn(x):
        return float(np.clip((hi - x) / (hi - lo) * 100.0, 0.0, 100.0))
    return _fn

pref_fn = linear_pref(f_lo, f_hi)
obj_names = ["f1", "f2"]

# 3. Configure stakeholders
preference_functions = {
    1: {"f1": pref_fn, "f2": pref_fn},
    2: {"f1": pref_fn, "f2": pref_fn},
}
objective_weights = {
    1: {"f1": 0.5, "f2": 0.5},
    2: {"f1": 0.5, "f2": 0.5},
}
stakeholder_weights = [0.3, 0.7]

# 4. Constraint-violation preference (deprioritises infeasible solutions)
def cv_pref_fn(cv):
    return float(100.0 / (1.0 + 10.0 * cv))

# 5. Run IGS-GA
algorithm = IGSGA(
    preference_functions=preference_functions,
    objective_weights=objective_weights,
    stakeholder_weights=stakeholder_weights,
    all_objectives=obj_names,
    pop_size=40,
    tournament_selection=True,
    archive=True,
    cv_pref_fn=cv_pref_fn,
    cv_weight=0.5,
    seed=42,
)

res = pymoo_minimize(problem, algorithm, get_termination("n_gen", 300), seed=42, copy_algorithm=False)

best   = res.algorithm.current_best
best_F = best.get("F").flatten()
print(f"Best found at gen {res.algorithm.current_best_gen}: f1={best_F[0]:.4f}, f2={best_F[1]:.4f}")
```

---

## `IGSGA` parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `preference_functions` | `dict` | `{stakeholder_id: {obj_name: callable}}` — maps objective values to preference scores (0–100) |
| `objective_weights` | `dict` | `{stakeholder_id: {obj_name: float}}` — per-stakeholder objective weights (must sum to 1 per stakeholder) |
| `stakeholder_weights` | `list[float]` | Relative importance of each stakeholder (normalised internally) |
| `all_objectives` | `list[str]` | Ordered list of objective names, matching the column order of `F` |
| `pop_size` | `int` | Population size (default `100`) |
| `tournament_selection` | `bool` | `True` (default): binary tournament; `False`: truncation selection |
| `archive` | `bool` | Maintain a cross-generation archive to stabilise IMAP z-score normalisation (default `False`) |
| `archive_max_size` | `int` | Maximum archive size before pruning (default `50`) |
| `cv_pref_fn` | `callable \| None` | Preference function for total constraint violation; `None` disables CV signal |
| `cv_weight` | `float` | Fraction of aggregation weight given to the CV signal, in `(0, 1)` (default `0.5`) |

### Stakeholder configuration pattern

```python
STAKEHOLDERS: dict[int, list[dict]] = {
    # Bi-objective problems
    2: [
        {"obj_weights": [0.5, 0.5], "weight": 0.3, "pref_shape": "linear"},
        {"obj_weights": [0.5, 0.5], "weight": 0.7, "pref_shape": "linear", "reverse": True},
    ],
    # Tri-objective problems
    3: [
        {"obj_weights": [1/3, 1/3, 1/3], "weight": 0.6, "pref_shape": "linear"},
        {"obj_weights": [0.2, 0.3, 0.5],  "weight": 0.4, "pref_shape": "linear", "reverse": True},
    ],
}
```

Each stakeholder dict supports:

| Key | Description |
|-----|-------------|
| `obj_weights` | Per-objective weight; must sum to 1 and have length `== n_obj` |
| `weight` | Importance of this stakeholder relative to others |
| `pref_shape` | `"linear"` \| `"convex"` \| `"concave"` \| `"sigmoid"` |
| `reverse` | If `True`, flips preference direction (higher objective value = better) |

---

## Result attributes

After calling `pymoo_minimize`, the result algorithm object exposes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `res.algorithm.current_best` | `Individual` | Best feasible solution found across all generations |
| `res.algorithm.current_best_gen` | `int` | Generation in which `current_best` was last updated |
| `res.algorithm.hall_of_fame` | `list[Individual]` | All previous `current_best` solutions, in order |
| `res.algorithm.best_archive` | `list[Individual]` | Archive used to stabilise IMAP scoring (when `archive=True`) |

---

## Preference shape reference

| Shape | Formula | Behaviour |
|-------|---------|-----------|
| `linear` | $p(t) = t \times 100$ | Proportional — default |
| `convex` | $p(t) = t^2 \times 100$ | Rewards proximity to optimum |
| `concave` | $p(t) = \sqrt{t} \times 100$ | Diminishing returns |
| `sigmoid` | $p(t) = \dfrac{100}{1 + e^{-10(t - 0.5)}}$ | Threshold / S-curve behaviour |

where $t = \dfrac{x_{\max} - x}{x_{\max} - x_{\min}} \in [0, 1]$ (higher $t$ = lower, better objective value).

---

## Demo notebook

`examples/das-cmop.ipynb` demonstrates the full workflow on the
[DAS-CMOP benchmark suite](https://pymoo.org/problems/multi/dascmop.html):

- Instantiate any `DASCMOP1`–`DASCMOP9` problem at a chosen difficulty level
- Derive analytical objective bounds with `get_das_cmop_bounds`
- Configure stakeholders using the `STAKEHOLDERS` dict
- Run IGS-GA and visualise convergence, Pareto front, and preference curves

---

## Benchmark

`benchmark_dascmop.py` compares IGS-GA against NSGA-II (bi-objective),
NSGA-III (tri-objective), C-TAEA, and MOEA/D-CDP across all 9 DAS-CMOP
problems × 16 difficulty levels, with N independent runs per configuration.
Results and plots are written to `tests/benchmark_dascmop_results/`.
