<div align="center">
  <h1>Optimizing the Cost-Quality of Agentic Theorem Provers in Lean</h1>

  <a href="https://www.python.org/">
    <img alt="Python: >=3.10" src="https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg">
  </a>
  <a href="https://github.com/astral-sh/uv">
    <img alt="Managed with uv" src="https://img.shields.io/badge/Package%20manager-uv-5c5c5c.svg">
  </a>
  <a href="./LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</div>

---

## Clone Repository

To clone the repository, one can run the following command:

```
git clone --recurse-submodules git@github.com:eth-sri/optimizing-lean-agents.git
```

## Quick Start

For this project, we use Lean version 4.9.0. To set up the environment, one can run the following commands. Note that the paths to the Lean math library and the API keys should be changed to the appropriate values for your system. The API key is needed for the data collection with the agent, but not for the whole-proof generation or the simulations.

```bash
# Python deps (agent = API clients, gpu = vLLM + torch)
uv sync --extra agent --extra gpu

# Lean
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y
source ~/.elan/env
cd mathlib4 && lake build && cd ..

# Env variables
export SCRATCH=./scratch && mkdir -p $SCRATCH/results
export TOGETHER_API_KEY=<your_key>
export CUDA_VISIBLE_DEVICES=<your-available-gpu>
```

To verify that the Lean environment is set up correctly, one can run the following command to check that the Lean math library compiles:

```
uv run python lean_compiler/repl_scheduler.py
```

## Data Generation

### Whole-Proof Generation
To run whole-proof generation, one can configure the run in `configs/hydra/prover/config.yaml` and then run it with

```
uv run python prover/runner.py --config configs/hydra/prover/config.yaml
```

### Agent
To run the agent data collection, one configures the run in `configs/hydra/seed_prover/config.yaml`, the prover module is configures in `configs/hydra/seed_prover/prover/unified.yaml`, and then

```
uv run python seed_prover/hydra_runner.py
```

## Simulations

Note that all the example configs are for the example problems in `datasets`. The simulations are based around the example train-test split in `dataset/example_problems_train_test_split.txt`. For reproducibility, we have included our version of the Putnam Lean formalizations along with our train-test splits in `dataset/putnam_rewrite_solved_train_test_split.txt`. In proof simulations, the train-test can be configured via `simulation.problem_split.path` in the config files.

### Our agent

To run the action routing Lean agent, we need three steps. Feature tracking, training the quality estimator model, and then running the sweep. The configs for these steps are in `configs/proof_simulation/example/`, example commands to run them are 

```
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/fixed_feature_tracker_full_router.yaml
uv run python scripts/proof_simulation/train_cost_logistic.py --config configs/proof_simulation/example/train_onestage.yaml
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/sweep_onestage.yaml
```

### Fixed-Step Baseline

To run the fixed-step baseline (and the whole-proof baseline to evaluate the performance of the data plane), one can adjust the configuration file found in `configs/proof_simulation/example/sweep_fixed_test.yaml` and then run the following command:

```
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/sweep_fixed_test.yaml
```

For whole-proof generation, one can run:

```
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/sweep_fixed_whole_proof.yaml
```

### Noisy Oracle Router

Similarly, to run the noisy oracle router, one can use the configs defined in `configs/proof_simulation/oracle` and run the feature tracker, training, and sweep steps exactly as described for our agent above.

## Graphical User Interface
We developed a GUI to analyze the data from the agent data collection and the simulations. To run the GUI, one can run the following command:

```
uv run --extra gui streamlit run analysis_gui/seed/app.py --server.headless=True
```

In the GUI, it is possible to both analyze the runs from the agent data collection step visually, and analyze trajectories and generate the plots from the simulations.