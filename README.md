# Deep Bellman Hedging in Hedging Gym

Thesis fork of [hedging-gym](https://github.com/0xC000005/hedging-gym), used to develop and
evaluate **Deep Bellman Hedging** (Buehler, Murray and Wood,
[arXiv:2207.00932](https://arxiv.org/abs/2207.00932v4)) as a baseline of the library, together
with a Lean 4 formalisation of the paper's theorems. Both contributions are merged upstream —
[PR #2](https://github.com/0xC000005/hedging-gym/pull/2) (Python, commit `15a562e`) and
[PR #3](https://github.com/0xC000005/hedging-gym/pull/3) (Lean, commit `a475769`) — and `main`
here tracks upstream. What this fork adds is the study record: the development branches
`deep-bellman-hedging` and `deep-bellman-formal`, and the release
[`dbh-study-2026-09-13`](https://github.com/spirituslab/hedging-gym/releases/tag/dbh-study-2026-09-13)
with the compact experiment artifacts.

## The contribution

| Piece | Where | What it does |
|---|---|---|
| `dbh` baseline | [`baselines/deep_bellman_hedging.py`](src/hedging_gym/baselines/deep_bellman_hedging.py) | Actor-critic value iteration under an optimized certainty equivalent: OCE utilities (entropic, CVaR, truncated entropy, Vicky, quadratic, identity) with the shift network, the Buehler-zero critic, random books sampled from the simulated bank, and one-step marked-wealth rewards that telescope to the ledger's terminal P&L. Options beyond the published scheme: `steps` (the paper's `T_n`), `scenarios` (conditional continuations), `aggregate="entropic"`. |
| Entropic objective | [`RiskConfig(objective="entropy")`](src/hedging_gym/environment/config.py) | Entropic risk as a terminal objective — the only time-consistent OCE besides the mean, hence the like-for-like comparison between nested and pathwise training; checked against pfhedge. |
| Qualification runner | [`benchmarks/qualify_deep_bellman.py`](benchmarks/qualify_deep_bellman.py) | Seeds × initial books, cached banks and delta tables, Deep Hedging and delta controls, reload and ledger reconciliation, `result.json` with revision and lockfile hash. |
| Lean proofs | [`formal/`](formal/README.md) | Theorem 1 (unique bounded fixed point via the monotone, cash-invariant contraction), value-iteration convergence, the finite-horizon case at β = 1, the vanilla Deep Hedging equation (Theorem 3), OCEs as monetary utilities (entropic and CVaR), the critic-loss gradient identity and the finite statistical-arbitrage bound. Lean 4.33.1 / mathlib v4.33.1, no `sorry`. |
| Docs and tests | [`docs/baseline-methods.md`](docs/baseline-methods.md#training-objectives-are-not-interchangeable), [`tests/test_deep_bellman_hedging.py`](tests/test_deep_bellman_hedging.py) | Mechanism notes, the nested-CVaR versus terminal-ES caveat and the estimator options; 42 numeric invariants (rewards telescope in every settlement mode, finite-difference actor gradient, exact CPU/CUDA resume, float32 utility normalisation, …). |

Differences from the paper are disclosed in the module docstring: the reward enters the utility as
in the paper's Definition 1, the critic uses the unconditional squared loss, cash is not a network
input, learner units are one-day stock moves, and books are simulated rather than tabulated.

## Findings

Seeds 7, 8, 9; fresh paired evaluation paths; mean ± std over seeds; lower is better.

| Task / objective | DBH | Deep Hedging | Delta |
|---|---|---|---|
| Heston default, entropic λ=10, published scheme, 5 000 updates | 0.00171 ± 0.00054 | 0.00060 ± 0.00002 | 0.00361 |
| same, DBH matched to Deep Hedging's sampled transitions | 0.00141 ± 0.00018 | — | — |
| same, `steps=10` / `steps=30` | 0.00065 ± 0.00008 / 0.00075 ± 0.00006 | 0.00060 | 0.00361 |
| GBM default, entropic λ=10, `steps=1/5/10/30` | 0.00260 ± 0.00129 / 0.00193 ± 0.00027 / 0.00171 ± 0.00007 / 0.00162 ± 0.00003 | 0.00162 ± 0.00009 | 0.00154 |
| GBM, `scenarios=8` / `32` / `32, aggregate="entropic"` | 0.00199 ± 0.00036 / 0.00273 ± 0.00104 / 0.00207 ± 0.00024 | — | — |
| Default ES95 task, CVaR utility | 0.026 ± 0.029 (one of three seeds diverged) | 0.00426 ± 0.00001 | 0.0613 |
| Bühler task (frictionless, unbounded), ES50 | 0.837 ± 0.009 | 0.955 ± 0.038 | — |

- DBH beats the delta hedge on the Heston tasks and pathwise Deep Hedging on the frictionless
  Bühler task; on the GBM task the delta hedge is best. With the risk-neutral utility the critic's
  initial value is ≈ 0, as the paper predicts for a book at its book value without statistical
  arbitrage.
- With proportional costs, the published one-step estimator stayed 2–3× behind Deep Hedging at
  every budget tried in this implementation; the paper's own n-step operator closes the gap
  (`steps=10` on Heston: 0.00065 versus 0.00060).
- Diagnostics on the entropic tasks: per-sample actor-gradient signal-to-noise 0.02–0.07; a
  critic 7–9× too pessimistic with an uninformative derivative in holdings; a policy that is right
  on average but scattered per state and trades twice as much; a deficit growing linearly with the
  remaining horizon. More conditional scenarios, more critic steps, wider networks or 30× longer
  training did not change this.
- The authors' numerical companion (Murray, Wood, Buehler, Wiese and Pakkanen,
  [ICAIF 2022](https://arxiv.org/abs/2207.07467)) reproduces vanilla Deep Hedging only with a
  Polyak target critic, an actor skip connection over the delta, a book-value critic residual,
  on-policy episodes and exponential-form losses — none of which the theory paper's scheme
  contains. Those ingredients, and the other estimator variants, are laid out in the follow-up
  section of [PR #2](https://github.com/0xC000005/hedging-gym/pull/2).
- The budget-matched run matches 38.4 M sampled transitions; DBH evaluates each pair for the actor
  and for the critic, so this is not a claim of identical compute.
- With the CVaR utility the critic represents the nested utility, which is not the evaluator's
  terminal expected shortfall; value checks are only comparable under the entropic utility.

## Reproduce

```bash
git clone https://github.com/spirituslab/hedging-gym.git && cd hedging-gym
uv sync --locked
uv run --frozen pytest tests/test_deep_bellman_hedging.py tests/test_config.py -q   # 42 tests, ~10 s
uv run --frozen python benchmarks/qualify_deep_bellman.py --objective entropy --risk-aversion 10 \
  --updates 100 --train-paths 1024 --eval-paths 2048 --seeds 7 --compare-dh \
  --device cuda --bank-dir /tmp/dbh-banks --output-dir /tmp/dbh-smoke   # ~6 min, mostly bank generation
```

The full study — task configurations, launch scripts, the 23 `result.json` files, summary tables,
diagnostic scripts and a reproduction README — is the release asset
[`dbh-study-2026-09-13.zip`](https://github.com/spirituslab/hedging-gym/releases/tag/dbh-study-2026-09-13);
`python3 summarize.py` inside it regenerates every table. Trade tapes and checkpoints are available
on request. Lean: `cd formal && lake exe cache get && lake build`.

## Hedging Gym

The library this work lives in: a configurable financial environment, classical and learned
baselines, and paper benchmark configurations. All baselines use the same market simulation,
observations, legal trades, cash ledger and terminal-loss evaluation.

| Area | What it provides | Guide |
|---|---|---|
| Environment | GBM, Heston and Bates markets; configurable instruments and execution; Gymnasium and PyTorch interfaces | [Getting started](docs/getting-started.md) |
| Baselines | Delta hedges, Deep Hedging, Deep Bellman Hedging, PPO, AlphaZero, adaptation and other named methods | [Baseline guide](docs/baselines.md) |
| Paper benchmarks | Bühler Heston and AlphaZero Heston/GBM configurations, runners and source comparisons | [Paper benchmarks](docs/paper-benchmarks.md) |

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed,
`uv sync --locked` builds the locked environment (Python dependencies and QuantLib) and
`uv run --frozen python -m hedging_gym.quickstart --device cpu --paths 64` runs scripted episodes
that illustrate the API. `uv run --frozen python -m benchmarks.baselines --methods dh ntb dbh`
runs a small comparison with classical controls; the defaults exercise the code and are not
paper-reproduction budgets — see [reproducibility](docs/baseline-implementation.md).

```text
src/hedging_gym/
  interfaces.py      public controller contract
  environment/       financial core, stepping, episodes and simulated branches
  adapters/          connectors to external learning libraries
  baselines/         one named entry point per complete method
    _shared/         reused learner mechanics, not additional methods
  extensions/        additions to existing policies and training procedures
  evaluation.py      common controller evaluation
benchmarks/          runnable comparisons and paper configuration files
tests/               financial, API and baseline checks
docs/                usage, sources and reproducible validation
formal/              standalone Lean proofs for Deep Bellman Hedging
```

Reference: [benchmark](docs/benchmark.md) defaults and equations,
[custom instruments](docs/custom-instruments.md), [validation](docs/validation.md),
[related work](docs/related-work.md), [contributing](CONTRIBUTING.md).

## License

Original code and documentation use the [MIT License](LICENSE). Incorporated third-party code
retains its own license; see [Third-party notices](THIRD_PARTY_NOTICES.md).
