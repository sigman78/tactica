"""``tactica`` CLI: play, tournament, noise-floor, skill-curve, sprt, replay."""
from __future__ import annotations

import argparse
import os
import sys
import time

from tactica.actions import Action
from tactica.battle import Battle
from tactica.eval.runner import (
    GameRecord,
    derive_seed,
    read_jsonl,
    run_match,
    run_pairs,
    replay_game,
    write_jsonl,
)
from tactica.eval.stats import mean_ci95, score_to_elo, sprt_bounds, sprt_llr
from tactica.eval.tournament import (
    format_matrix,
    format_ratings,
    format_scenario_breakdown,
    openskill_ratings,
    run_tournament,
)
from tactica.scenario import load_scenario, resolve_scenarios


def _result_line(record: GameRecord) -> str:
    outcome = "draw" if record.winner is None else f"side {record.winner} ({record.specs[record.winner]}) wins"
    return (f"{record.scenario_name} seed={record.seed}: {outcome} "
            f"after {record.rounds} rounds, hash={record.state_hash}")


# ----------------------------------------------------------------------- #
# play


def cmd_play(args: argparse.Namespace) -> int:
    scenario = load_scenario(args.scenario, deterministic=args.deterministic or None)
    on_step = None
    if args.render:
        def on_step(battle: Battle, action: Action) -> None:
            print(f"\n>>> {action!r}")
            print(battle.render())
        print(Battle.from_scenario(scenario, args.seed).render())
    record = run_match(args.p0, args.p1, scenario, args.seed, on_step=on_step)
    print(_result_line(record))
    if args.out:
        write_jsonl(args.out, [record.to_dict()])
        print(f"replay written to {args.out}")
    return 0


# ----------------------------------------------------------------------- #
# tournament


def cmd_tournament(args: argparse.Namespace) -> int:
    specs = [s.strip() for s in args.agents.split(",") if s.strip()]
    if len(specs) < 2:
        print("need at least two agents", file=sys.stderr)
        return 2
    scenarios = resolve_scenarios(args.scenarios)
    t0 = time.perf_counter()
    result = run_tournament(specs, scenarios, args.pairs, args.seed,
                            workers=args.workers, out_path=args.out,
                            progress=True)
    dt = time.perf_counter() - t0
    n_games = len(result.games)
    print(f"\n{n_games} games ({n_games // 2} mirrored pairs) in {dt:.1f}s\n")
    print(format_matrix(result))
    print()
    print(format_scenario_breakdown(result))
    print()
    print(format_ratings(openskill_ratings(result)))
    if args.out:
        print(f"\ngame log written to {args.out}")
    return 0


# ----------------------------------------------------------------------- #
# noise-floor


def _game_score0(game: dict) -> float:
    winner = game["winner"]
    return 0.5 if winner is None else (1.0 if winner == 0 else 0.0)


def cmd_noise_floor(args: argparse.Namespace) -> int:
    scenarios = resolve_scenarios(args.scenarios)
    print(f"Noise floor: {args.agent} vs itself, {args.pairs} mirrored pairs "
          f"per scenario (CRN seeds, base={args.seed})\n")
    all_pair: list[float] = []
    all_game: list[float] = []
    for sc in scenarios:
        tasks = [(args.agent, args.agent, sc.to_dict(),
                  derive_seed(args.seed, sc.name, i)) for i in range(args.pairs)]
        pair_scores: list[float] = []
        game_scores: list[float] = []
        for g1, g2, score in run_pairs(tasks, workers=args.workers):
            pair_scores.append(score)
            game_scores.extend((_game_score0(g1), _game_score0(g2)))
        all_pair.extend(pair_scores)
        all_game.extend(game_scores)
        gm, gci = mean_ci95(game_scores)
        pm, pci = mean_ci95(pair_scores)
        print(f"  {sc.name:>20}: game-level side-0 score={gm:.4f} +/-{gci:.4f}"
              f" (dev {gm - 0.5:+.4f})   pair-level={pm:.4f} +/-{pci:.4f}")
    gm, gci = mean_ci95(all_game)
    pm, pci = mean_ci95(all_pair)
    print(f"\n  {'OVERALL':>20}: game-level side-0 score={gm:.4f} +/-{gci:.4f}"
          f" (dev {gm - 0.5:+.4f})   pair-level={pm:.4f} +/-{pci:.4f}")
    print(
        "\nGame-level deviation from 0.5 is the luck floor for UNPAIRED"
        "\ncomparisons (side advantage + sampling noise). Pair-level scores"
        "\nshow what mirrored pairs + CRN buy you: for a deterministic agent"
        "\nthe two games of a pair are exact mirrors and the noise is zero;"
        "\nany apparent skill difference smaller than these bands is luck."
    )
    return 0


# ----------------------------------------------------------------------- #
# skill-curve


def cmd_skill_curve(args: argparse.Namespace) -> int:
    epsilons = [float(e) for e in args.epsilons.split(",")]
    scenarios = resolve_scenarios(args.scenarios)
    print(f"Skill curve: epsilon({args.agent}, eps) vs {args.agent}, "
          f"{args.pairs} pairs x {len(scenarios)} scenarios per point\n")
    rows: list[tuple[float, float, float]] = []
    for eps in epsilons:
        spec = args.agent if eps == 0 else f"epsilon:{eps:g}:{args.agent}"
        scores: list[float] = []
        for sc in scenarios:
            tasks = [(spec, args.agent, sc.to_dict(),
                      derive_seed(args.seed, sc.name, i))
                     for i in range(args.pairs)]
            scores.extend(s for _, _, s in run_pairs(tasks, workers=args.workers))
        mean, ci = mean_ci95(scores)
        rows.append((eps, mean, ci))
        print(f"  eps={eps:<5g} score={mean:.4f} +/-{ci:.4f}")

    print("\n(0.5 = noise mistakes cost nothing; lower = decisions matter)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot "
              "(uv sync, or pip install tactica[eval])")
        return 0
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("epsilon (random-move probability)")
    ax.set_ylabel(f"pair score vs clean {args.agent}")
    ax.set_title("Skill curve: cost of random mistakes")
    ax.set_ylim(0, 0.6)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120)
    print(f"plot saved to {args.plot}")
    return 0


# ----------------------------------------------------------------------- #
# sprt


def cmd_sprt(args: argparse.Namespace) -> int:
    candidate = f"weighted:{args.candidate}"
    baseline = f"weighted:{args.baseline}"
    scenarios = resolve_scenarios(args.scenarios)
    lower, upper = sprt_bounds(args.alpha, args.beta)
    print(f"SPRT: H1 elo>={args.elo1} vs H0 elo<={args.elo0}, "
          f"alpha={args.alpha} beta={args.beta}")
    print(f"LLR bounds: accept H0 at {lower:.3f}, accept H1 at {upper:.3f}\n")

    wins = draws = losses = 0
    trajectory: list[float] = []
    verdict = "inconclusive"
    pair_index = 0
    while pair_index < args.max_pairs:
        sc = scenarios[pair_index % len(scenarios)]
        seed = derive_seed(args.seed, sc.name, pair_index)
        tasks = [(candidate, baseline, sc.to_dict(), seed)]
        ((g1, g2, _score),) = tuple(run_pairs(tasks, workers=1))
        for g, cand_side in ((g1, 0), (g2, 1)):
            if g["winner"] is None:
                draws += 1
            elif g["winner"] == cand_side:
                wins += 1
            else:
                losses += 1
        llr = sprt_llr(wins, draws, losses, args.elo0, args.elo1)
        trajectory.append(llr)
        pair_index += 1
        n = wins + draws + losses
        if pair_index % 25 == 0:
            print(f"  n={n:5d}  WDL={wins}/{draws}/{losses}  LLR={llr:+.3f}")
        if llr >= upper:
            verdict = "accept H1 (candidate is stronger)"
            break
        if llr <= lower:
            verdict = "accept H0 (no improvement)"
            break

    n = wins + draws + losses
    score = (wins + draws / 2) / n if n else 0.5
    print(f"\nVerdict: {verdict}")
    print(f"Games: {n} ({pair_index} mirrored pairs)  WDL={wins}/{draws}/{losses}  "
          f"score={score:.4f} (~{score_to_elo(score):+.1f} elo)")
    print(f"Final LLR: {trajectory[-1]:+.3f}  (bounds [{lower:.3f}, {upper:.3f}])")
    tail = ", ".join(f"{x:+.2f}" for x in trajectory[-12:])
    print(f"LLR trajectory (last 12 of {len(trajectory)} pairs): {tail}")
    return 0


# ----------------------------------------------------------------------- #
# replay


def cmd_replay(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.file)
    if not rows:
        print("empty replay file", file=sys.stderr)
        return 2
    if not 0 <= args.index < len(rows):
        print(f"index {args.index} out of range (file has {len(rows)} games)",
              file=sys.stderr)
        return 2
    record = GameRecord.from_dict(rows[args.index])

    on_step = None
    if args.render:
        def on_step(battle: Battle, action: Action) -> None:
            print(f"\n>>> {action!r}")
            print(battle.render())
    battle = replay_game(record, on_step=on_step)
    print(_result_line(record))
    print(f"replay OK: final state hash matches ({battle.state_hash()})")
    return 0


# ----------------------------------------------------------------------- #
# parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tactica", description="HoMM-style tactics AI sandbox")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, scenarios_default: str = "all") -> None:
        p.add_argument("--seed", type=int, default=1, help="base seed")
        p.add_argument("--scenarios", default=scenarios_default,
                       help="'all', a name, a JSON path, or a comma list")
        p.add_argument("--workers", type=int, default=0,
                       help="worker processes (0 = one per CPU core, 1 = inline)")

    p = sub.add_parser("play", help="run one battle")
    p.add_argument("--p0", required=True, help="agent spec for side 0")
    p.add_argument("--p1", required=True, help="agent spec for side 1")
    p.add_argument("--scenario", default="open_field")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--render", action="store_true", help="ASCII board per turn")
    p.add_argument("--deterministic", action="store_true",
                   help="expected-value damage, no chance nodes")
    p.add_argument("--out", help="write the game record (replayable JSONL)")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("tournament", help="round-robin over mirrored pairs")
    p.add_argument("--agents", required=True, help="comma list of agent specs")
    p.add_argument("--pairs", type=int, default=50,
                   help="mirrored pairs per matchup per scenario")
    p.add_argument("--out", help="JSONL output path for all games")
    common(p)
    p.set_defaults(func=cmd_tournament)

    p = sub.add_parser("noise-floor", help="agent vs itself: the luck baseline")
    p.add_argument("--agent", default="heuristic")
    p.add_argument("--pairs", type=int, default=500)
    common(p)
    p.set_defaults(func=cmd_noise_floor)

    p = sub.add_parser("skill-curve", help="WR degradation vs epsilon noise")
    p.add_argument("--agent", default="heuristic")
    p.add_argument("--epsilons", default="0,0.05,0.1,0.2,0.5")
    p.add_argument("--pairs", type=int, default=300,
                   help="pairs per scenario per epsilon")
    p.add_argument("--plot", default="skill_curve.png")
    common(p)
    p.set_defaults(func=cmd_skill_curve)

    p = sub.add_parser("sprt", help="sequential test: candidate vs baseline weights")
    p.add_argument("--candidate", required=True, help="candidate weights JSON")
    p.add_argument("--baseline", required=True, help="baseline weights JSON")
    p.add_argument("--elo0", type=float, default=0.0)
    p.add_argument("--elo1", type=float, default=10.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--max-pairs", type=int, default=5000)
    common(p)
    p.set_defaults(func=cmd_sprt)

    p = sub.add_parser("replay", help="re-simulate a logged game, verify hash")
    p.add_argument("--file", required=True, help="JSONL game log")
    p.add_argument("--index", type=int, default=0, help="game row to replay")
    p.add_argument("--render", action="store_true")
    p.set_defaults(func=cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "workers", 1) == 0:
        args.workers = os.cpu_count() or 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
