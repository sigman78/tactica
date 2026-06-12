# Evaluation methodology

Tactics battles are noisy: damage rolls, initiative tiebreaks, and asymmetric
maps can fake or bury an agent improvement. The tooling attacks variance
from several directions:

- **Mirrored pairs.** The unit of play is two games on the same scenario and
  seed with sides swapped. Map asymmetry and side advantage cancel within
  the pair instead of inflating variance across the sample.
- **Common random numbers (CRN).** Pair `i` on scenario `s` uses seed
  `derive_seed(base_seed, s, i)` (a stable sha256 derivation — Python's
  salted `hash()` is never used) for *every* matchup in a tournament. All
  agents face the same battles, so matchup comparisons are paired and much
  tighter than independent sampling at the same game count.
- **Noise floor.** Before believing "A beats B by 3%", run `noise-floor`:
  an agent against itself "should" score 0.500, and the measured deviation
  with its CI is the resolution limit of your experiment. The pair-level
  numbers also demonstrate what mirroring buys: deterministic self-play
  pairs are exact mirrors, so their paired noise is zero.
- **Skill curve.** `skill-curve` injects an `eps` rate of random moves and
  measures the cost. It calibrates how much decisions matter on these maps —
  if eps=0.2 barely moved the score, you couldn't expect agent improvements
  to show up either. (Here a 5% blunder rate already costs ~3 points of
  score and 50% costs ~29, so decisions matter.)
- **SPRT.** For iterating on `WeightedAgent` parameters, the sequential
  probability ratio test streams mirrored pairs and stops as soon as the
  evidence crosses the alpha/beta bounds — usually far earlier than a
  fixed-n experiment with the same error guarantees.

Every game is logged as one JSONL row (scenario, seed, agent configs, action
list, winner, rounds, final state hash), and `tactica replay` re-simulates
any row and asserts the hash — reproducibility is enforced, not assumed.

The CLI commands implementing all of this are documented in [cli.md](cli.md).
