"""
IMAP-GA: pymoo-native IMAP affine aggregation genetic algorithm.

Replaces NSGA-II's non-dominated sorting and crowding distance with a two-pass
affine aggregation selection (IMAP) using the existing a_fine_aggregator kernel.

Classes
-------
IMAPSurvival : Survival
    Two-pass IMAP survival selection.
IGSGA : NSGA2
    Algorithm class with current_best / hall_of_fame tracking.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.survival import Survival
from pymoo.operators.selection.tournament import TournamentSelection, compare

from optimization._aggregator import a_fine_aggregator

# ---------------------------------------------------------------------------
# Tournament comparator
# ---------------------------------------------------------------------------

def imap_tournament_selection(
    pop: Population,
    P: np.ndarray,
    algorithm: "IGSGA",
    random_state: np.random.Generator | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Binary tournament selection based on IMAP preference scores.

    Falls back to random selection when ``imap_score`` has not been set yet
    (e.g. during the very first mating call before any survival has run).

    Called by :class:`~pymoo.operators.selection.tournament.TournamentSelection`
    as ``func_comp(pop, P, random_state=random_state, **kwargs)``.  The
    ``algorithm`` keyword arrives via the ``**kwargs`` propagated from the
    mating pipeline.

    Parameters
    ----------
    pop:
        The current population.
    P:
        Integer array of shape ``(n_tournaments, 2)`` with competitor indices.
    algorithm:
        The :class:`IGSGA` instance (provides access to ``current_best``).
    random_state:
        Seeded generator for reproducible tie-breaking.

    Returns
    -------
    np.ndarray
        Flat 1-D integer array of winning indices, shape ``(n_tournaments,)``.
        ``TournamentSelection._do`` reshapes the result itself.
    """
    n_tournaments, n_parents = P.shape
    if n_parents != 2:
        raise ValueError("imap_tournament only supports binary (pressure=2) tournaments.")

    S = np.full(n_tournaments, np.nan)

    for i in range(n_tournaments):
        a, b = P[i, 0], P[i, 1]
        score_a = pop[a].get("imap_score") # type: ignore
        score_b = pop[b].get("imap_score") # type: ignore

        if score_a is None or score_b is None:
            # imap_score not yet assigned — random selection as fallback
            S[i] = random_state.choice([a, b]) # type: ignore
            continue

        # CDP: if is_feasible was stored by IMAPSurvival, a feasible
        # individual always beats an infeasible one regardless of score.
        feas_a = pop[a].get("is_feasible") # type: ignore
        feas_b = pop[b].get("is_feasible") # type: ignore

        if feas_a is not None and feas_b is not None:
            fa, fb = bool(feas_a), bool(feas_b)
            if fa and not fb:
                S[i] = a
                continue
            elif fb and not fa:
                S[i] = b
                continue
            # Both same feasibility → fall through to score comparison

        S[i] = compare(
            a, score_a,
            b, score_b,
            method="larger_is_better",
            return_random_if_equal=True,
            random_state=random_state,
        )

    return S.astype(int)


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------

class IMAPSurvival(Survival):
    """
    Two-pass IMAP affine aggregation survival selection.

    Implements the selection procedure described in Sections 2-3 of the
    IMAP-GA specification:

    1. First-pass aggregation on the full pool to produce scores in [0, 100].
    2. Discard individuals scoring ≤ 40 (skipped if fewer than *N* would remain).
    3. Second-pass aggregation on the surviving candidates.
    4. Update ``algorithm.current_best`` and ``algorithm.hall_of_fame``.
    5. Return the top *N* individuals, guaranteeing ``current_best`` is kept.
    """

    def __init__(self, algorithm: "IGSGA") -> None:
        """
        Parameters
        ----------
        algorithm:
            The parent :class:`IGSGA` instance.  Used as a fallback when
            ``algorithm`` is not passed through ``**kwargs`` to :meth:`_do`.
        """
        super().__init__(filter_infeasible=False)
        self.algorithm = algorithm

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_objective_values(self, pop: Population) -> dict[str, list[float]]:
        """
        Build the ``objective_values`` dict expected by
        :meth:`IGSGA.get_aggregated_scores`.

        When ``algorithm.cv_pref_fn`` is set a special ``"cv"`` key is added,
        containing the total constraint violation for each individual (computed
        from ``G`` as ``sum(max(0, g_j))``).

        Parameters
        ----------
        pop:
            Pool of evaluated individuals.

        Returns
        -------
        dict[str, list[float]]
            Maps each objective name to a list of raw values in pool order.
            Includes ``"cv"`` when a CV preference function is active.
        """
        obj_vals: dict[str, list[float]] = {
            key: [] for key in self.algorithm.all_objectives
        }
        use_cv = self.algorithm.cv_pref_fn is not None
        if use_cv:
            obj_vals["cv"] = []

        for i in range(len(pop)):
            for key, val in zip(self.algorithm.all_objectives, pop[i].get("F")):
                obj_vals[key].append(float(val))
            if use_cv:
                G = pop[i].get("G")
                cv = float(np.sum(np.maximum(0.0, G))) if G is not None else 0.0
                obj_vals["cv"].append(cv)
        return obj_vals

    # ------------------------------------------------------------------
    # Core selection
    # ------------------------------------------------------------------

    def _do(
        self,
        problem,
        pop: Population,
        *args,
        n_survive: int | None = None,
        **kwargs,
    ) -> Population:
        """
        Execute two-pass IMAP selection on *pop* and return *N* survivors.

        Parameters
        ----------
        problem:
            The pymoo problem (not used directly; objective values come from
            evaluated ``pop[i].get("F")``).
        pop:
            Combined pool: parents + offspring + optionally injected
            ``current_best``.
        n_survive:
            Target survivor count.  Falls back to ``len(algorithm.pop)``
            if ``None``.
        **kwargs:
            Must contain ``algorithm`` (:class:`IGSGA` instance) and
            ``random_state``.

        Returns
        -------
        Population
            *N* selected survivors, or all of *pop* if the pool is smaller
            than *N*.
        """
        alg: IGSGA = kwargs.get("algorithm", self.algorithm)
        N: int = n_survive if n_survive is not None else len(alg.pop)

        # --- CDP setup ---
        # Active whenever the problem has constraints; no-op for unconstrained
        # problems (is_feasible is all-True, so every CDP branch is identity).
        use_cdp: bool = problem.n_constr > 0
        if use_cdp:
            G = pop.get("G")
            cv = (
                np.sum(np.maximum(0.0, G), axis=1)
                if G is not None
                else np.zeros(len(pop))
            )
            is_feasible = cv <= 1e-9          # shape (len(pop),)
            n_feasible_total = int(is_feasible.sum())
        else:
            is_feasible = np.ones(len(pop), dtype=bool)
            n_feasible_total = len(pop)

        # --- Step 1: First-pass aggregation ---
        obj_vals = self._build_objective_values(pop)
        scores = alg.get_aggregated_scores(obj_vals)

        # --- Step 2: Discard phase (threshold = 40, feasibility-aware) ---
        keep_mask = scores > 40.0
        if use_cdp and n_feasible_total < N:
            # Case A: fewer feasible solutions than N →
            # protect every feasible solution regardless of its score.
            keep_mask |= is_feasible
        # Case B (n_feasible >= N): threshold applies to all — no override.

        n_natural_keep = int(keep_mask.sum())
        cb_x = alg.current_best.get("X") if alg.current_best is not None else None

        if n_natural_keep < N:
            candidates = pop
            is_feasible_cands = is_feasible
        else:
            # Force-keep current_best only when it is feasible (or when no
            # feasible solutions exist in the pool — fallback to old behaviour).
            if cb_x is not None:
                cb_is_feasible = False
                for idx, ind in enumerate(pop):
                    if np.array_equal(ind.get("X"), cb_x): # type: ignore
                        cb_is_feasible = bool(is_feasible[idx])
                        break
                if not use_cdp or cb_is_feasible or n_feasible_total == 0:
                    for idx, ind in enumerate(pop):
                        if np.array_equal(ind.get("X"), cb_x): # type: ignore
                            keep_mask[idx] = True
                            break
            candidates = pop[keep_mask]
            is_feasible_cands = is_feasible[keep_mask]

        # --- Step 3: Second-pass aggregation ---
        # When archive is enabled, archive members are appended to the objective
        # values dict before scoring.  This stabilises the Z-score reference frame
        # across generations.  Archive scores are computed but immediately discarded;
        # only the first len(candidates) scores are kept.
        if len(candidates) >= 2:
            obj_vals2 = self._build_objective_values(candidates) # type: ignore
            n_candidates = len(candidates)
            if alg.use_archive and len(alg.best_archive) > 0:
                for ind in alg.best_archive:
                    for key, val in zip(alg.all_objectives, ind.get("F")): # type: ignore
                        obj_vals2[key].append(float(val))
                    if "cv" in obj_vals2:
                        G = ind.get("G")
                        cv = float(np.sum(np.maximum(0.0, G))) if G is not None else 0.0 # type: ignore
                        obj_vals2["cv"].append(cv)
            all_scores = alg.get_aggregated_scores(obj_vals2)
            final_scores = all_scores[:n_candidates]
        else:
            # Edge case: 0 or 1 survivor — skip second pass
            final_scores = np.full(
                len(candidates),
                100.0 if len(candidates) == 1 else 50.0,
            )

        # --- Step 4: Store imap_score (and is_feasible for CDP tournament) ---
        candidates.set("imap_score", final_scores) # type: ignore
        if use_cdp:
            candidates.set("is_feasible", is_feasible_cands) # type: ignore

        # --- Step 5: Update current_best / hall_of_fame ---
        # With CDP: prefer the highest-scoring feasible candidate.
        # Fall back to overall best only when no feasible candidates exist.
        n_feasible_cands = int(is_feasible_cands.sum())
        if use_cdp and n_feasible_cands > 0:
            feasible_idx = np.where(is_feasible_cands)[0]
            best_local_idx = int(
                feasible_idx[np.argmax(final_scores[feasible_idx])]
            )
        else:
            best_local_idx = int(np.argmax(final_scores))
        best_local: Individual = candidates[best_local_idx]
        best_local_score: float = float(final_scores[best_local_idx])

        if alg.current_best is None:
            alg.current_best = best_local
            alg.current_best_gen = alg._gen
        else:
            # Find current_best's score in this generation's second-pass context.
            # current_best was injected into the pool and kept through the discard
            # phase, so it should be present in candidates with a fresh final_score
            # that is directly comparable to best_local_score.
            cb_x = alg.current_best.get("X")
            cb_score_this_gen: float | None = None
            for idx, ind in enumerate(candidates):
                if np.array_equal(ind.get("X"), cb_x): # type: ignore
                    cb_score_this_gen = float(final_scores[idx])
                    break

            is_different = not np.array_equal(cb_x, best_local.get("X")) # type: ignore
            # Replace only when best_local is strictly better in this generation's
            # scoring context AND is a genuinely different solution.
            is_better = (
                cb_score_this_gen is None          # current_best was discarded
                or best_local_score > cb_score_this_gen
            )
            if is_different and is_better:
                if alg.use_archive:
                    alg._add_to_archive(alg.current_best)
                alg.hall_of_fame.append(alg.current_best)
                alg.current_best = best_local
                alg.current_best_gen = alg._gen
            # else: no change to current_best or hall_of_fame

        # --- Step 6: Select N survivors; guarantee current_best survives ---
        if len(candidates) <= N:
            return candidates # type: ignore

        # Locate current_best in candidates by X-array equality (needed for
        # the force-keep guarantee in both selection modes below).
        cb_x_final = alg.current_best.get("X")
        cb_in_candidates: int | None = next(
            (
                i for i, ind in enumerate(candidates)
                if np.array_equal(ind.get("X"), cb_x_final) # type: ignore
            ),
            None,
        )

        if alg.tournament_selection:
            # ── Tournament selection ───────────────────────────────────────────
            # Run N binary tournaments on the candidates pool.  Each tournament
            # samples two candidates uniformly at random (with replacement).
            # CDP rules apply: feasible always beats infeasible; within the same
            # feasibility class the higher final_score wins.
            rng: np.random.Generator = kwargs.get(
                "random_state", np.random.default_rng()
            )
            n_cands = len(candidates)
            selected: list[int] = []

            for _ in range(N):
                a = int(rng.integers(0, n_cands))
                b = int(rng.integers(0, n_cands))
                if a == b:
                    selected.append(a)
                    continue
                if use_cdp:
                    fa, fb = bool(is_feasible_cands[a]), bool(is_feasible_cands[b])
                    if fa and not fb:
                        selected.append(a)
                        continue
                    elif fb and not fa:
                        selected.append(b)
                        continue
                # Same feasibility (or no CDP): higher score wins; random tie-break.
                if final_scores[a] > final_scores[b]:
                    selected.append(a)
                elif final_scores[b] > final_scores[a]:
                    selected.append(b)
                else:
                    selected.append(int(rng.choice([a, b])))

            if cb_in_candidates is not None and cb_in_candidates not in selected:
                selected[-1] = cb_in_candidates

            return candidates[np.array(selected)] # type: ignore

        else:
            # ── Truncation selection ─────────────────────────
            if use_cdp:
                # Strictly take top-N feasible first, fill remaining
                # slots from top infeasible by IMAP score.
                feas_idx   = np.where(is_feasible_cands)[0]
                infeas_idx = np.where(~is_feasible_cands)[0]
                feas_sorted = (
                    feas_idx[np.argsort(final_scores[feas_idx])[::-1]]
                    if len(feas_idx) > 0 else np.empty(0, dtype=int)
                )
                infeas_sorted = (
                    infeas_idx[np.argsort(final_scores[infeas_idx])[::-1]]
                    if len(infeas_idx) > 0 else np.empty(0, dtype=int)
                )
                n_feas_take   = min(N, len(feas_sorted))
                n_infeas_take = N - n_feas_take
                top_n = (
                    list(feas_sorted[:n_feas_take])
                    + list(infeas_sorted[:n_infeas_take])
                )
            else:
                sort_idx = np.argsort(final_scores)[::-1]
                top_n = list(sort_idx[:N])

            if cb_in_candidates is not None and cb_in_candidates not in top_n:
                top_n[-1] = cb_in_candidates  # replace last slot with current_best

            return candidates[np.array(top_n)] # type: ignore


# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

class IGSGA(NSGA2):
    """
    IMAP-GA: NSGA-II variant with IMAP affine aggregation survival.

    Replaces non-dominated sorting and crowding distance entirely with a
    two-pass IMAP preference aggregation (:class:`IMAPSurvival`).  Maintains
    a cross-generation ``current_best`` individual and a ``hall_of_fame`` list.

    Parameters
    ----------
    preference_functions:
        Outer key: stakeholder ID (int, 1-indexed).
        Inner key: objective name (str).
        Value: callable mapping one raw objective value → preference score in [0, 100].
    objective_weights:
        Per-stakeholder weight per objective.  Should sum to 1 per stakeholder.
    stakeholder_weights:
        One weight per stakeholder (normalised internally; need not sum to 1).
    all_objectives:
        Ordered list of objective names matching column order of
        ``individual.get("F")``.
    pop_size:
        Population size. Default 100.
    tournament_selection : bool
        Controls how the N survivors are chosen from the post-discard candidates.

        ``True`` *(default)* — tournament: run N binary tournaments on the candidates pool.
        Each tournament samples two candidates at random; the higher-scoring
        (and, under CDP, feasible-preferred) one wins.  Lower-scoring candidates
        retain a non-zero chance of surviving, which preserves more genetic
        diversity at the cost of weaker selection pressure.

        ``False`` — truncation: deterministically keep the top-N
        candidates by IMAP score (respecting CDP ordering)
    archive : bool
        When ``True``, maintains an archive of previously-best solutions that
        are included in the second-pass IMAP scoring pool to stabilise Z-score
        normalisation across generations.  Archive members participate in
        scoring only; they are never returned as survivors.  Default ``False``.
    archive_max_size : int
        Maximum number of individuals held in the archive before pruning.
        Once the archive exceeds this limit, IMAP is run over all archive
        members and the bottom 50 % are discarded.  Default ``50``.
    **kwargs:
        Forwarded to :class:`~pymoo.algorithms.moo.nsga2.NSGA2` /
        :class:`~pymoo.algorithms.base.genetic.GeneticAlgorithm`
        (e.g. ``seed``, ``verbose``).
    """

    def __init__(
        self,
        preference_functions: dict[int, dict[str, Callable[[float], float]]],
        objective_weights: dict[int, dict[str, float]],
        stakeholder_weights: list[float],
        all_objectives: list[str],
        pop_size: int = 100,
        tournament_selection: bool = True,
        archive: bool = False,
        archive_max_size: int = 50,
        cv_pref_fn: Callable[[float], float] | None = None,
        cv_weight: float = 0.5,
        **kwargs,
    ) -> None:
        # Store IMAP configuration BEFORE calling super() so that
        # IMAPSurvival (instantiated below) can reference self.all_objectives.
        self.preference_functions = preference_functions
        self.objective_weights = objective_weights
        self.stakeholder_weights = stakeholder_weights
        self.all_objectives = all_objectives

        self.tournament_selection: bool = tournament_selection
        self.use_archive: bool = archive
        self.best_archive_max_size: int = archive_max_size

        # CV preference signal (optional).
        # When set, cv_weight fraction of the total aggregation weight is given
        # to this function applied to each individual's total constraint violation.
        # All stakeholder weights are scaled to (1 - cv_weight) collectively.
        if cv_pref_fn is not None and not (0.0 < cv_weight < 1.0):
            raise ValueError(f"cv_weight must be in (0, 1), got {cv_weight}.")
        self.cv_pref_fn: Callable[[float], float] | None = cv_pref_fn
        self.cv_weight: float = cv_weight

        # Tracking structures — overwritten by _initialize_advance
        self.current_best: Individual | None = None
        self.current_best_gen: int = 0   # generation in which current_best was last set
        self.hall_of_fame: list[Individual] = []
        self.best_archive: list[Individual] = []
        self._gen: int = 0               # internal counter: 0 = init pop, 1+ = regular gens

        _survival = IMAPSurvival(self)
        _selection = TournamentSelection(func_comp=imap_tournament_selection, pressure=2)

        super().__init__(
            pop_size=pop_size,
            selection=_selection,
            survival=_survival, # type: ignore
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Aggregation (mirrors BRKGA.get_aggregated_scores verbatim)
    # ------------------------------------------------------------------

    def get_aggregated_scores(
        self, objective_values: dict[str, list[float]]
    ) -> np.ndarray:
        """
        Compute IMAP preference scores for a pool of individuals.

        Delegates to :func:`~allodyn.optimization.a_fine_aggregator.a_fine_aggregator`
        via the same two-layer stakeholder / objective pipeline used in
        ``BRKGA.get_aggregated_scores``.

        Parameters
        ----------
        objective_values:
            Maps each objective name to a list of raw values, one per
            individual in the pool, in pool order.

        Returns
        -------
        np.ndarray
            1-D array of aggregated preference scores in [0, 100].
        """
        self.stakeholders = self.preference_functions.keys()
        total_weight = sum(self.stakeholder_weights)
        s_weights = [w / total_weight for w in self.stakeholder_weights]

        p: list[list[float]] = []
        w: list[float] = []

        for stakeholder in self.stakeholders:
            for objective in self.all_objectives:
                fn = self.preference_functions.get(stakeholder, {}).get(
                    objective, lambda x: 0
                )
                p.append([fn(x) for x in objective_values[objective]])
                w.append(
                    self.objective_weights[stakeholder][objective]
                    * s_weights[stakeholder - 1]
                )

        # ── Optional CV signal ──────────────────────────────────────────────
        # When a CV preference function is configured, scale all stakeholder
        # weights down to (1 - cv_weight) and append the CV criterion with
        # weight cv_weight.  The total then sums to 1 as required.
        # When the population is fully feasible all CV scores equal 100, giving
        # std ≈ 0; a_fine_aggregator replaces that with 1e-6 so the CV term
        # contributes essentially nothing — objective quality decides the rank.
        if self.cv_pref_fn is not None and "cv" in objective_values:
            scale = 1.0 - self.cv_weight
            w = [wi * scale for wi in w]
            w.append(self.cv_weight)
            p.append([self.cv_pref_fn(cv) for cv in objective_values["cv"]])

        return a_fine_aggregator(w, p) # type: ignore

    # ------------------------------------------------------------------
    # Archive management
    # ------------------------------------------------------------------

    def _add_to_archive(self, individual: Individual) -> None:
        """
        Add *individual* to the archive if it is not already present.

        Deduplication is based on X-array equality.  If the archive exceeds
        ``archive_max_size`` after insertion, :meth:`_prune_archive` is called
        immediately to trim it back to 50 % capacity.

        Parameters
        ----------
        individual:
            The individual to archive (typically the outgoing ``current_best``).
        """
        x_new = individual.get("X")
        for existing in self.best_archive:
            if np.array_equal(existing.get("X"), x_new): # type: ignore
                return   # already present — skip
        self.best_archive.append(individual)
        if len(self.best_archive) > self.best_archive_max_size:
            self._prune_archive()

    def _prune_archive(self) -> None:
        """
        Score all archive members against each other and keep the top 50 %.

        Runs :meth:`get_aggregated_scores` over the archive in isolation
        (archive members only) and discards the lower-scoring half.  At least
        one individual is always retained.
        """
        if len(self.best_archive) < 2:
            return
        obj_vals: dict[str, list[float]] = {key: [] for key in self.all_objectives}
        for ind in self.best_archive:
            for key, val in zip(self.all_objectives, ind.get("F")): # type: ignore
                obj_vals[key].append(float(val))
        scores = self.get_aggregated_scores(obj_vals)
        n_keep = max(1, len(self.best_archive) // 2)
        top_indices = np.argsort(scores)[::-1][:n_keep]
        self.best_archive = [self.best_archive[i] for i in top_indices]

    # ------------------------------------------------------------------
    # pymoo lifecycle hooks
    # ------------------------------------------------------------------

    def _initialize_advance(
        self, infills: Population | None = None, **kwargs
    ) -> None:
        """
        Post-initialisation hook: set up ``current_best`` and ``hall_of_fame``.

        Called once by :meth:`~pymoo.core.algorithm.Algorithm.advance` after
        the initial population has been evaluated.  Delegates all scoring and
        tracking logic to :class:`IMAPSurvival._do` in a single consistent
        pass.

        .. warning::
            Do **not** call ``super()._initialize_advance()`` — that would
            invoke ``RankAndCrowdingSurvival`` and overwrite ``imap_score``
            values.
        """
        self.hall_of_fame = []    # starts EMPTY per spec; must be set BEFORE survival.do
        self.best_archive = []         # cleared on each fresh run
        self.current_best = None  # _do will set it when it identifies best_idx
        self.pop = self.survival.do( # type: ignore
            self.problem,
            infills,
            n_survive=len(infills), # type: ignore
            algorithm=self,
            random_state=self.random_state,
        )

    def _advance(self, infills: Population | None = None, **kwargs) -> None:
        """
        Per-generation advance: merge pool, inject ``current_best``, apply survival.

        Does **not** call ``super()._advance()`` to avoid triggering
        ``RankAndCrowdingSurvival``.

        Parameters
        ----------
        infills:
            Evaluated offspring generated by ``_infill()``.
        """
        assert infills is not None, "_advance called without infills."

        self._gen += 1

        # Merge current population with offspring
        pool = Population.merge(self.pop, infills)

        # Inject current_best only if not already present (elite elitism guarantee)
        if self.current_best is not None:
            cb_x = self.current_best.get("X")
            already_in = any(
                np.array_equal(cb_x, pool[i].get("X")) # type: ignore
                for i in range(len(pool)) # type: ignore
            )
            if not already_in:
                pool = Population.merge(pool, Population.create(self.current_best))

        # Apply IMAP survival (two-pass scoring, incumbent update, top-N selection)
        self.pop = self.survival.do( # type: ignore
            self.problem,
            pool,
            n_survive=self.pop_size,
            algorithm=self,
            random_state=self.random_state,
        )

    def _set_optimum(self, **kwargs) -> None:
        """
        Set ``self.opt`` to the IMAP-tracked incumbent (``current_best``).

        Overrides NSGA-II's rank-0 based optimum selection so that
        ``res.opt`` reflects the best solution found across all generations.
        """
        if self.current_best is not None:
            self.opt = Population.create(self.current_best)
        elif self.pop is not None and len(self.pop) > 0:
            scores = self.pop.get("imap_score")
            if scores is not None:
                self.opt = self.pop[[int(np.argmax(scores))]]
            else:
                warnings.warn(
                    "_set_optimum called before imap_score was set on population; "
                    "falling back to first individual. This should only occur in "
                    "edge cases.",
                    stacklevel=2,
                )
                self.opt = self.pop[[0]]
