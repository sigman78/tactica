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
    simulations (128 sims ~ 0.25 vs 8 sims on open_field). Uninformed
    rollouts undervalue initiating melee trades (the attacker always eats
    retaliation; the tempo payoff is beyond the evaluation horizon), so
    extra search converges on that bias instead of fixing it. Better
    rollout policies or tree search are the obvious next experiments —
    this sandbox exists to make those measurable.

See [extending.md](extending.md) for adding your own agent, and
[evaluation.md](evaluation.md) for how to measure it honestly.
