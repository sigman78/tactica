"""Soft strength-regression test: agent win-rates vs. stored baselines.

A fixed ``(agents, scenarios, pairs, seed)`` tournament is fully deterministic
-- common random numbers plus seeded agents, and weighted/heuristic carry no
RNG -- so the pair-scores below are reproducible on a given code version and
move only when *behavior* changes. This complements the byte-exact
replay/determinism tests: a change can be perfectly deterministic yet make an
agent weaker, which those tests cannot see and this one can.

Banded by design: each matchup must stay within ``TOL`` of its baseline and the
ladder ordering (weighted > heuristic > random) must hold. Small, intentional
behavior changes that stay inside the band do not require action.

Re-blessing: when you change agent/engine behavior on purpose and a band trips,
first confirm the new numbers are an improvement (``tactica sprt`` /
``tactica tournament``), then update ``BASELINES``. Print fresh numbers with:

    .venv/Scripts/python.exe tests/test_strength_regression.py
"""
from __future__ import annotations

import pytest

from tactica.eval.tournament import run_tournament
from tactica.scenario import BUILTIN_SCENARIOS

# Explicit scenario list (not "all") so adding a builtin scenario later does not
# silently shift the baselines.
AGENTS = ["weighted", "heuristic", "random"]
SCENARIOS = ["open_field", "skirmish", "archers_vs_cavalry"]
PAIRS = 15
SEED = 7
TOL = 0.05

# Pair-score of (row vs col), captured on the directional-melee branch. Exact
# for this config; re-bless on intentional behavior changes (see module docs).
BASELINES: dict[tuple[str, str], float] = {
    ("weighted", "heuristic"): 0.622,
    ("weighted", "random"): 0.889,
    ("heuristic", "random"): 0.911,
}


def _matrix() -> dict[tuple[str, str], tuple[float, float, int]]:
    scenarios = [BUILTIN_SCENARIOS[n] for n in SCENARIOS]
    return run_tournament(AGENTS, scenarios, PAIRS, SEED, workers=1).matrix()


@pytest.fixture(scope="module")
def matrix() -> dict[tuple[str, str], tuple[float, float, int]]:
    return _matrix()


@pytest.mark.parametrize(("a", "b"), list(BASELINES))
def test_pair_score_within_band(matrix, a, b) -> None:
    score = matrix[(a, b)][0]
    expected = BASELINES[(a, b)]
    assert abs(score - expected) <= TOL, (
        f"{a} vs {b}: {score:.3f} drifted from baseline {expected:.3f} "
        f"(+/-{TOL}). If the change was intentional, re-bless BASELINES.")


def test_ladder_ordering_holds(matrix) -> None:
    wh = matrix[("weighted", "heuristic")][0]
    hr = matrix[("heuristic", "random")][0]
    wr = matrix[("weighted", "random")][0]
    assert wh > 0.5, f"weighted no longer beats heuristic ({wh:.3f})"
    assert hr > 0.5, f"heuristic no longer beats random ({hr:.3f})"
    assert wr > 0.5, f"weighted no longer beats random ({wr:.3f})"


if __name__ == "__main__":  # bless: print fresh baselines to copy in
    m = _matrix()
    print("Current pair-scores (copy into BASELINES):")
    for pair in BASELINES:
        print(f'    ("{pair[0]}", "{pair[1]}"): {m[pair][0]:.3f},')
