"""Round-robin tournament over mirrored pairs with common random numbers."""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

from tactica.eval.runner import derive_seed, run_pairs, write_jsonl
from tactica.eval.stats import mean_ci95
from tactica.scenario import Scenario


@dataclass
class TournamentResult:
    agents: list[str]
    # pair scores per (a, b) matchup, a's perspective, across all scenarios
    scores: dict[tuple[str, str], list[float]] = field(default_factory=dict)
    by_scenario: dict[tuple[str, str, str], list[float]] = field(default_factory=dict)
    games: list[dict] = field(default_factory=list)

    def matrix(self) -> dict[tuple[str, str], tuple[float, float, int]]:
        """(mean pair score, 95% CI half-width, n pairs) per ordered matchup."""
        out = {}
        for (a, b), vals in self.scores.items():
            mean, ci = mean_ci95(vals)
            out[(a, b)] = (mean, ci, len(vals))
            out[(b, a)] = (1.0 - mean, ci, len(vals))
        return out


def run_tournament(
    agent_specs: list[str],
    scenarios: list[Scenario],
    pairs: int,
    base_seed: int,
    workers: int = 1,
    out_path: str | None = None,
    progress: bool = False,
) -> TournamentResult:
    result = TournamentResult(agents=list(agent_specs))
    tasks: list[tuple[str, str, dict, int]] = []
    meta: list[tuple[str, str, str, int]] = []
    for a, b in itertools.combinations(agent_specs, 2):
        for sc in scenarios:
            for i in range(pairs):
                # CRN: the seed depends only on (base_seed, scenario, i) so
                # every matchup plays the same battles.
                seed = derive_seed(base_seed, sc.name, i)
                tasks.append((a, b, sc.to_dict(), seed))
                meta.append((a, b, sc.name, i))

    for k, (g1, g2, score) in enumerate(run_pairs(tasks, workers=workers)):
        a, b, sc_name, i = meta[k]
        for g, mirrored in ((g1, False), (g2, True)):
            g["pair_index"] = i
            g["mirrored"] = mirrored
            result.games.append(g)
        result.scores.setdefault((a, b), []).append(score)
        result.by_scenario.setdefault((a, b, sc_name), []).append(score)
        if progress and (k + 1) % 25 == 0:
            print(f"  ... {k + 1}/{len(tasks)} pairs done", flush=True)

    if out_path:
        write_jsonl(out_path, result.games)
    return result


def openskill_ratings(result: TournamentResult) -> list[tuple[str, float, float, float]] | None:
    """Rate agents from per-game outcomes. Returns rows of
    (agent, mu, sigma, ordinal) sorted by ordinal, or None if openskill
    is not installed."""
    try:
        from openskill.models import PlackettLuce
    except ImportError:
        return None
    model = PlackettLuce()
    ratings = {a: model.rating(name=a) for a in result.agents}
    for g in result.games:
        s0, s1 = g["specs"]
        winner = g["winner"]
        ranks = [1, 1] if winner is None else ([1, 2] if winner == 0 else [2, 1])
        [[r0], [r1]] = model.rate([[ratings[s0]], [ratings[s1]]], ranks=ranks)
        ratings[s0], ratings[s1] = r0, r1
    rows = [(a, r.mu, r.sigma, r.ordinal()) for a, r in ratings.items()]
    rows.sort(key=lambda r: -r[3])
    return rows


def format_matrix(result: TournamentResult) -> str:
    agents = result.agents
    matrix = result.matrix()
    width = max(10, max(len(a) for a in agents) + 1)
    lines = ["Pair-score matrix (row vs column, 0.5 = even):"]
    header = " " * width + "".join(f"{a:>{width}}" for a in agents)
    lines.append(header)
    for a in agents:
        cells = []
        for b in agents:
            if a == b:
                cells.append(f"{'-':>{width}}")
            else:
                mean, ci, _ = matrix[(a, b)]
                cells.append(f"{f'{mean:.3f}±{ci:.2f}':>{width}}")
        lines.append(f"{a:>{width}}" + "".join(cells))
    return "\n".join(lines)


def format_scenario_breakdown(result: TournamentResult) -> str:
    lines = ["Per-scenario pair scores (matchup: scenario=score):"]
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for (a, b, sc), vals in sorted(result.by_scenario.items()):
        mean, ci = mean_ci95(vals)
        grouped[(a, b)].append(f"{sc}={mean:.3f}±{ci:.2f}")
    for (a, b), cells in grouped.items():
        lines.append(f"  {a} vs {b}: " + "  ".join(cells))
    return "\n".join(lines)


def format_ratings(rows: list[tuple[str, float, float, float]] | None) -> str:
    if rows is None:
        return ("OpenSkill ratings skipped: package not installed "
                "(uv sync, or pip install tactica[eval]).")
    lines = ["OpenSkill ratings (PlackettLuce; ordinal = mu - 3*sigma):"]
    for agent, mu, sigma, ordinal in rows:
        lines.append(f"  {agent:>24}  mu={mu:7.3f}  sigma={sigma:6.3f}  "
                     f"ordinal={ordinal:7.3f}  95% mu-range=[{mu - 1.96 * sigma:.2f}, "
                     f"{mu + 1.96 * sigma:.2f}]")
    return "\n".join(lines)
