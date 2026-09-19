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

## Clone the Repository

Clone the repository together with its `mathlib4` submodule:

```bash
git clone --recurse-submodules git@github.com:eth-sri/optimizing-lean-agents.git
```

## Install Lean

This project uses Lean 4.9.0. Install Elan and build the checked-in `mathlib4`
submodule with:

```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh -s -- -y
source ~/.elan/env
cd mathlib4 && lake build && cd ..
```

To verify that the Lean environment is set up correctly, one can run the following command to check that the Lean math library compiles:

```bash
uv run python lean_compiler/repl_scheduler.py
```

The output should include the following line if the Lean environment is set up correctly:

```text
Progress: 1/1 proofs processed. REPL errors: 0
```

## Quick Start

Install the API-based agent dependencies and create the output directory:

```bash
uv sync --extra agent

export SCRATCH=./scratch
mkdir -p "$SCRATCH/results"
```

Set the API key required by the model configuration you select, such as
`TOGETHER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or
`GOOGLE_API_KEY`.

For local model serving, Linux hosts with CUDA can additionally run:

```bash
uv sync --extra agent --extra gpu
export CUDA_VISIBLE_DEVICES=<your-available-gpu>
```

The `gpu` extra contains vLLM and does not resolve on macOS.
The provided data-generation configs select local vLLM models by default, so
use the GPU setup on a compatible host or change their model configs to an
API-backed provider before running them.

## Data Generation

### Whole-Proof Generation

Configure the run in `configs/hydra/prover/config.yaml`, then execute:

```bash
uv run python prover/runner.py --config configs/hydra/prover/config.yaml
```

### Agent

Configure agent data collection in `configs/hydra/seed_prover/config.yaml` and
the inner prover in `configs/hydra/seed_prover/prover/unified.yaml`, then run:

```bash
uv run python seed_prover/hydra_runner.py
```

## Simulations

All example configs use the problems in `dataset/`. The simulations use the
train/test split in `dataset/example_problems_train_test_split.txt`. For
reproducibility, the repository also includes our Putnam Lean formalizations
and split in `dataset/putnam_rewrite_solved_train_test_split.txt`. Configure a
simulation split with `simulation.problem_split.file` and
`simulation.problem_split.split`.

The simulation configs consume the artifacts produced by the whole-proof and
agent data-generation steps above. Their default paths are under
`scratch/results/whole_proof_example_8b` and
`scratch/results/data_plane_example`, respectively.

### Our agent

Run feature tracking, train the quality estimator, and then run the router
sweep:

```bash
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/fixed_feature_tracker_full_router.yaml
uv run python scripts/proof_simulation/train_cost_logistic.py --config configs/proof_simulation/example/train_onestage.yaml
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/sweep_onestage.yaml
```

### Fixed-Step Baseline

Run the fixed-step baseline with:

```bash
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/sweep_fixed_test.yaml
```

For whole-proof generation, one can run:

```bash
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/example/sweep_fixed_whole_proof.yaml
```

### Noisy Oracle Router

The noisy-oracle pipeline likewise has three steps:

```bash
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/oracle/fixed_feature_tracker_oracle_train.yaml
uv run python scripts/proof_simulation/train_logistic.py --config configs/proof_simulation/oracle/train_oracle_logit_train.yaml
uv run python scripts/proof_simulation/sweep.py --config configs/proof_simulation/oracle/sweep_noisy_oracle.yaml
```

## Graphical User Interface

We developed a GUI to analyze the data from the agent data collection and the simulations. To run the GUI, one can run the following command:

```bash
uv run --extra agent --extra gui streamlit run analysis_gui/seed/app.py --server.headless=true
```

In the GUI, it is possible to both analyze the runs from the agent data collection step visually, and analyze trajectories and generate the plots from the simulations.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{rögnvaldsson2026optimizingcostqualitytradeoffagentic,
      title={Optimizing the Cost-Quality Tradeoff of Agentic Theorem Provers in Lean},
      author={Kári Rögnvaldsson and Chenhao Sun and Jasper Dekoninck and Martin Vechev},
      year={2026},
      eprint={2606.04883},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2606.04883},
}
```
