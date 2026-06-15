# Agent ladder

Agent specs accepted by every CLI command and the dashboard: `random`,
`heuristic`, `weighted[:weights.json]`, `mcts[:SIMS[:C_UCT[:ROLLOUT_CAP]]]`,
`epsilon:EPS:INNER`.

- **random** — uniform over legal actions; the floor every agent must beat.
- **heuristic** — scripted: archers focus the lowest-HP target and kite when
  cornered; melee attacks the highest-value reachable target or advances on
  it; defends when nothing is useful. The strongest shipped agent.
- **weighted** — scores every legal action with a linear feature vector
  (expected damage dealt/received, kill bonus, target value, focus fire,
  distance terms, wait/defend flags) and plays the argmax. The shipped
  default weights are an aggressive profile that was SPRT-accepted at
  ~+11.5 elo over the original heuristic-imitating weights
  (`weights/conservative.json`).
- **epsilon:EPS:INNER** — plays a uniform random legal action with
  probability EPS, otherwise defers to INNER. The measurement probe behind
  `skill-curve`.
- **mcts** — flat UCT with several documented twists that earned their place
  through measurements you can rerun:
  - *CRN-paired rounds*: simulations are allocated in full rounds where
    every arm shares the round's rollout seed, so seed luck is common-mode
    and cancels from the ranking. (Unpaired UCB revisits let the max over
    ~50 one-sample means be won by lucky outlier arms — measurably, 8-sim
    agents beat 128-sim agents until this change.)
  - *Progressive widening*: only ~sqrt(2·sims) arms are considered, attack
    arms first — rollout evaluations rank attacks usefully but are noisy
    and systematically retreat-happy on movement arms.
  - *Rollout-length discount*: values decay ~30% over the rollout cap, so a
    kill now beats the same win 100 plies later. Without it the agent rolls
    out a won position from every arm, ties at +1.0, and dawdles forever.
  - *Short biased rollouts*: `Battle.playout` (attack- and chase-biased
    random policy) capped at 40 steps by default, then the material-balance
    eval. The spec's literal 200-step cap is available as `mcts:S:C:200`,
    but it measures weaker (0.42 +/- 0.12 head-to-head vs the 40-step
    default, and 0.22 vs heuristic instead of 0.35) and runs ~5x slower:
    long random rollouts drown the root action's effect in outcome noise.
  - A known, measured limitation: strength does **not** scale cleanly with
    simulations. Head-to-head vs the heuristic: 32 sims -> 0.30, 512 -> 0.42,
    1024 -> 0.25 (small samples, but flat-to-down, not up). This is **not** a
    two-player sign/perspective bug -- flat UCT has no minimax backup to
    negate, the rollout value is read consistently from the root player's
    side, and `mcts` beats `random` 0.83 (a sign-bugged search could not).
    The ceiling is the evaluator: `Battle.playout` models both sides as
    biased-random, so the search optimizes "best vs random play" (great vs
    random, weak vs a competent heuristic), and the material-only leaf is
    position-blind. Extra search sharpens that biased objective instead of
    fixing it.
  - *Heuristic-guided rollouts* — `MCTSAgent(rollout_policy="heuristic")`, an
    epsilon-greedy `HeuristicAgent` rollout for both sides — are the lever,
    not more sims: at just **64 sims it beats the heuristic 0.71** on the same
    open_field+skirmish config where random-rollout `mcts:512` managed 0.42.
    Cost is ~10s/game (the heuristic runs every rollout step), so it is a
    strength experiment, not the default, and is currently constructor-only
    (not in the `mcts:...` spec). Next: tune epsilon/sims, SPRT it, and try a
    positional leaf eval. This sandbox exists to make those measurable.

See [extending.md](extending.md) for adding your own agent, and
[evaluation.md](evaluation.md) for how to measure it honestly.
