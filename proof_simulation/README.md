# proof_simulation

Simulation framework that replays pre-recorded proof data under different policies to evaluate proof search strategies. No actual proving — it draws from historical attempt data.

## File Structure

```
proof_simulation/
├── actions.py           # Action, ActionType, DetailedCost, ActionResult
├── target.py            # TargetNode — core recursive abstraction (~500 lines)
├── breakdown_state.py   # Round-robin queue within a breakdown
├── problem.py           # SimulatedProblem — wrapper around root TargetNode
├── state.py             # SimulationState — read-only snapshot for policies
├── trajectory.py        # Trajectory recording (step = state + action + result)
├── simulation.py        # SimulationRunner — orchestrates policy over problems
├── analysis.py          # Aggregate stats (solve rate, cost distribution, build_seed_summary)
├── config.py            # YAML config loading, build_policy(), construction
├── features/            # Composable StateTracker features (oracle, noisy_oracle,
│                        #   avg_cost, normalized_similarity, error_diversity,
│                        #   subgoal_repetition, code_preview, predicted_prob)
├── data/
│   ├── types.py         # AttemptData, AttemptPair, BreakdownTemplate
│   ├── full_proof.py    # Load minified full-proof JSONs (256 attempts/problem)
│   ├── agent.py         # Load seed prover Session data (breakdowns + corrections)
│   └── loader.py        # Unified factory merging data sources → SimulatedProblems
└── policies/
    ├── base.py          # Abstract Policy: choose_action(state, valid_actions) → Action
    ├── fixed.py         # FixedPolicy (staged waterfall with per-target budgets)
    ├── cost_models.py   # RunningAverageCostModel
    ├── cost_quality.py  # CostQualityPolicy: argmax [p(a) - λ·c(a)], pluggable models
    └── quality/         # Probability models: oracle, pretrained_logistic,
                         #   trajectory_logistic (+ logit / 1/x feature transforms)
```

The `run.py` / `sweep.py` CLIs and the experiment driver scripts live in
`scripts/proof_simulation/`; their configs live in `configs/proof_simulation/`.

## Core Abstractions

### TargetNode (the heart)

Recursive class representing any provable entity (problem, theorem, lemma). Forms a tree:

```
TargetNode (root = problem)
  ├── PROVE(model) → pops next pre-recorded attempt for that model
  ├── CORRECT → pops next correction from last failed AttemptPair
  └── DECOMPOSE → CREATE_BREAKDOWN → spawns child TargetNodes
       ├── theorem (id=-1)
       ├── lemma 0, lemma 1, ...
       └── managed by BreakdownState (round-robin queue)
```

Each node holds `{model: [AttemptPair]}` proof data and `[BreakdownTemplate]` for decomposition. `get_current_focus()` walks down to the active leaf.

### Actions

| Action | What it does |
|--------|-------------|
| `PROVE(model)` | Draw next attempt for model |
| `CORRECT` | Draw correction from last failed attempt |
| `DECOMPOSE` | Mark target as decomposed |
| `CREATE_BREAKDOWN` | Instantiate next breakdown template |
| `TERMINATE` | Give up |

### Simulation Loop

```
policy.choose_action(state, valid_actions) → action
problem.simulate_action(action) → result
trajectory.add_step(state, action, result)
```

Seed controls attempt shuffle order. Multi-seed gives statistical variation.

### Data Sources

- **Full proof**: Minified JSON, 256 attempts/problem, no breakdowns. Goes to root node's proof data.
- **Agent**: Seed prover Session with breakdowns, lemmas, corrections. Provides breakdown templates + per-target proof data.

`loader.py` merges both. Multiple agent sources can be merged by `breakdown_key`.

## How to Add Things

### New Policy

1. Create `policies/my_policy.py` implementing `Policy.choose_action(state, valid_actions) → Action`
2. Export in `policies/__init__.py`
3. Register in `config.py`'s `POLICY_REGISTRY` (or handle in `build_policy()` for special cases like oracle)
4. Reference as `policy.type: "my_policy"` in YAML config

### New Data Source

1. Create loader in `data/` returning `Dict[str, List[AttemptPair]]` or `Dict[str, List[BreakdownTemplate]]`
2. Wire into `data/loader.py`'s `load_problems()`
3. Add path to YAML config under `data:`

### New Action Type

1. Add to `ActionType` enum in `actions.py`
2. Handle in `TargetNode.execute_action()` and `get_valid_actions()`
3. Update `SimulatedProblem._handle_breakdown_update()` if it affects breakdown flow
4. Update policies

### New State Fields

Add to `SimulationState` in `state.py`, populate in `SimulatedProblem.get_state()` / `TargetNode.get_state_snapshot()`.

## Running

```bash
# Fixed-policy sweep over the example test split
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/sweep_fixed_test.yaml

# Cost/quality router sweep
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/sweep_onestage.yaml
```

### Included experiment configs

| Experiment | Command |
|---|---|
| 8B fixed-budget agent baseline | `sweep.py --config configs/proof_simulation/example/sweep_fixed_test.yaml` |
| 8B whole-proof baseline | `sweep.py --config configs/proof_simulation/example/sweep_fixed_whole_proof.yaml` |
| Oracle router | `sweep.py --config configs/proof_simulation/oracle/sweep_noisy_oracle.yaml` |
| One-stage cost/quality router | `sweep.py --config configs/proof_simulation/example/sweep_onestage.yaml` |

## Key Gotchas

- Seed shuffles attempt order, doesn't change the data itself
- `success` = `pass AND complete` (no sorries)
- Policies should be stateless (receive state, return action)
