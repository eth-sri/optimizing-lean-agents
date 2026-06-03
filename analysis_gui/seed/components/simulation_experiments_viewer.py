"""
Simulation Experiments Viewer

View and analyze results from simulation experiment runs with hyperparameter sweeps.
Displays ranking of hyperparameter combinations and drill-down into individual seeds.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
from datetime import datetime
import plotly.graph_objects as go

# Add root directory to path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(root_dir))


def load_summary_csv_or_json(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load summary data from CSV or JSON format."""
    csv_path = run_dir / "summary.csv"
    json_path = run_dir / "summary.json"

    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            st.error(f"Error loading {csv_path}: {e}")
            return None

    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Convert JSON to DataFrame format
            rows = []
            for problem_id, stats in data.items():
                rows.append({
                    "problem_id": problem_id,
                    "num_seeds_solved": stats.get("seeds_solved", 0),
                    "solve_rate": stats.get("seeds_solved", 0) / stats.get("total_seeds", 1) if stats.get("total_seeds", 0) > 0 else 0,
                    "avg_prover_calls": stats.get("avg_attempts", stats.get("avg_prover_calls", 0)),
                    "avg_sflops": stats.get("avg_sflops", 0),
                    "avg_used_lemmas": stats.get("avg_used_lemmas", 0),
                })
            return pd.DataFrame(rows)
        except Exception as e:
            st.error(f"Error loading {json_path}: {e}")
            return None

    return None


def get_simulation_runs(run_dir: Path) -> List[Path]:
    """
    Get list of simulation experiment timestamps in a run directory.

    Args:
        run_dir: Path to the run directory

    Returns:
        List of simulation experiment directories sorted by timestamp
    """
    simulations_dir = run_dir / "simulations"
    if not simulations_dir.exists():
        return []

    # Find all timestamp-based directories (YYYY/MM/DD/HHMMSS pattern)
    experiments = []
    for year_dir in simulations_dir.glob("20*"):
        if year_dir.is_dir():
            for month_dir in year_dir.glob("*"):
                if month_dir.is_dir():
                    for day_dir in month_dir.glob("*"):
                        if day_dir.is_dir():
                            for time_dir in day_dir.glob("*"):
                                if time_dir.is_dir():
                                    experiments.append(time_dir)

    # Sort by newest first
    return sorted(experiments, reverse=True)


def load_experiment_results(experiment_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load summary.csv for an experiment.

    Groups by hyperparameter combination (if subdirectories exist) or loads directly.

    Args:
        experiment_dir: Path to the experiment directory

    Returns:
        Dict mapping param_string -> DataFrame from summary.csv
    """
    results = {}

    # Check if summary.csv exists directly in experiment_dir
    summary_csv = experiment_dir / "summary.csv"
    if summary_csv.exists():
        try:
            df = pd.read_csv(summary_csv)
            results[experiment_dir.name] = df
            return results
        except Exception as e:
            st.error(f"Error loading {summary_csv}: {e}")

    # Otherwise, look for parameter subdirectories
    for param_dir in sorted(experiment_dir.iterdir()):
        if not param_dir.is_dir():
            continue

        param_str = param_dir.name
        summary_csv = param_dir / "summary.csv"

        if summary_csv.exists():
            try:
                df = pd.read_csv(summary_csv)
                results[param_str] = df
            except Exception as e:
                st.error(f"Error loading {summary_csv}: {e}")

    return results


def compute_aggregate_stats(summary_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Compute aggregate statistics from summary.csv.

    Args:
        summary_df: DataFrame from summary.csv
        config: Optional config dict to get num_seeds from

    Returns:
        Dict with aggregated statistics
    """
    if summary_df.empty:
        return {}

    total_problems = len(summary_df)
    problems_solved = summary_df["num_seeds_solved"].gt(0).sum()

    # Get num_seeds from config (source of truth)
    if config and "simulation" in config:
        num_seeds = config["simulation"].get("num_seeds", 16)
    else:
        # Fallback: infer from solve_rate if config not available
        num_seeds = None
        for _, row in summary_df.iterrows():
            if row["solve_rate"] > 0:
                num_seeds = int(row["num_seeds_solved"] / row["solve_rate"])
                break
        if num_seeds is None:
            num_seeds = 16  # Default fallback

    # Aggregate stats
    total_output_sflops = summary_df["avg_sflops"].sum()
    avg_output_sflops = total_output_sflops / problems_solved if problems_solved > 0 else 0

    total_prover_calls = summary_df["avg_prover_calls"].sum()
    avg_prover_calls = total_prover_calls / problems_solved if problems_solved > 0 else 0

    avg_used_lemmas = summary_df["avg_used_lemmas"].mean() if "avg_used_lemmas" in summary_df.columns else 0

    return {
        "num_seeds": num_seeds,
        "problems_solved": problems_solved,
        "problems_solved_count": problems_solved,
        "total_problems": total_problems,
        "avg_output_sflops": avg_output_sflops,
        "avg_prover_calls": avg_prover_calls,
        "avg_used_lemmas": avg_used_lemmas,
    }


def parse_param_string(param_str: str) -> Dict[str, Any]:
    """
    Parse parameter string back into a dict.

    Format: "confidence_threshold=5_explore_steps=3"

    Args:
        param_str: Parameter string from directory name

    Returns:
        Dict with parsed parameters
    """
    params = {}
    for part in param_str.split("_"):
        if "=" in part:
            key, value = part.split("=", 1)
            # Try to convert to int, otherwise keep as string
            try:
                params[key] = int(value)
            except ValueError:
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value

    return params


def translate_strategy_name(strategy: str) -> str:
    """
    Translate strategy name to readable version.

    Args:
        strategy: Full strategy name

    Returns:
        Readable strategy name
    """
    strategy_map = {
        "Simple": "Greedy",
        "Probabilistic": "Probabilistic",
        "ProofLengthPredictor": "Proof Length",
        "ReasoningTrace": "Reasoning Trace",
    }
    return strategy_map.get(strategy, strategy)


def prettify_config(param_str: str, strategy: Optional[str] = None) -> str:
    """
    Prettify config parameters for display.

    Args:
        param_str: Parameter string from directory name
        strategy: Optional strategy name to include

    Returns:
        Prettified config string
    """
    params = parse_param_string(param_str)

    # Build pretty string with key abbreviations
    key_map = {
        "confidence_threshold": "conf",
        "threshold": "t",
        "auto_fit": "auto",
        "explore_steps": "explore",
        "max_iterations": "maxiter",
        "max_lemmas": "maxlem",
        "use_8b": "8b",
        "use_32b": "32b",
    }

    # Skip irrelevant None parameters
    skip_if_none = {"attempts", "iter", "max_attempts", "max_iter"}

    parts = []
    if strategy:
        parts.append(translate_strategy_name(strategy))

    # Prioritize model size params (use_8b, use_32b) - show first
    priority_keys = ["use_8b", "use_32b"]
    for key in priority_keys:
        if key in params:
            value = params[key]
            short_key = key_map.get(key, key)
            if isinstance(value, bool) or value in ["True", "False"]:
                if value == True or value == "True":
                    parts.append(short_key)
            elif value not in ["None", None]:
                parts.append(f"{short_key}={value}")

    # Then add other params
    for key, value in sorted(params.items()):
        if key in priority_keys:
            continue  # Already added

        # Skip None values for irrelevant params
        if key in skip_if_none and (value == "None" or value is None):
            continue

        # Skip all None values
        if value == "None" or value is None:
            continue

        short_key = key_map.get(key, key)
        # Format booleans nicely
        if isinstance(value, bool) or value in ["True", "False"]:
            if value == True or value == "True":
                parts.append(short_key)
        else:
            parts.append(f"{short_key}={value}")

    return " | ".join(parts) if parts else param_str


def render_multi_run_comparison_tab(base_run_dir: Path):
    """Render the multi-run comparison view."""
    st.header("📊 Multi-Run Comparison")

    st.markdown("""
    Compare multiple simulation runs side-by-side. Select runs to compare and choose a metric.
    The best performer for each problem is highlighted in **green**.
    """)

    # Find base results directory
    results_base = Path("../../scratch/results")
    if not results_base.exists():
        results_base = Path("../scratch/results")
    if not results_base.exists():
        results_base = Path("scratch/results")

    dump_base = Path("../../scratch/dump")
    if not dump_base.exists():
        dump_base = Path("../scratch/dump")
    if not dump_base.exists():
        dump_base = Path("scratch/dump")

    # Find all simulation runs with summary files
    all_runs = []

    # Helper to load strategy from config
    def get_strategy(run_path: Path) -> Optional[str]:
        config_path = run_path / "config.yaml"
        # If config doesn't exist in current dir, try parent (for hyperparameter subdirs)
        if not config_path.exists():
            config_path = run_path.parent / "config.yaml"

        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return config.get("strategy", {}).get("type")
            except:
                pass
        return None

    # Add simulations from current base_run_dir
    if base_run_dir:
        simulations_dir = base_run_dir / "simulations"
        if simulations_dir.exists():
            for year_dir in simulations_dir.glob("20*"):
                for month_dir in year_dir.glob("*"):
                    for day_dir in month_dir.glob("*"):
                        for time_dir in day_dir.glob("*"):
                            if (time_dir / "summary.csv").exists() or (time_dir / "summary.json").exists():
                                timestamp = f"{year_dir.name}/{month_dir.name}/{day_dir.name} {time_dir.name}"
                                strategy = get_strategy(time_dir)
                                strategy_short = translate_strategy_name(strategy) if strategy else ""
                                display = f"{timestamp} - {strategy_short}" if strategy_short else timestamp
                                all_runs.append((display, time_dir))

                            # Check hyperparameter subdirectories
                            for param_dir in time_dir.iterdir():
                                if param_dir.is_dir() and ((param_dir / "summary.csv").exists() or (param_dir / "summary.json").exists()):
                                    timestamp = f"{year_dir.name}/{month_dir.name}/{day_dir.name} {time_dir.name}"
                                    strategy = get_strategy(param_dir)
                                    config_pretty = prettify_config(param_dir.name, strategy)
                                    display = f"{timestamp} - {config_pretty}"
                                    all_runs.append((display, param_dir))

    if not all_runs:
        st.warning("No simulation runs with summary files found in this directory.")
        return

    # Baseline paths
    baseline_8b = results_base / "full_proof_8b/2025/12/13/simulations_1024"
    baseline_32b = results_base / "full_proof_32b/2025/12/13/simulations_1024"

    # Multi-select for runs
    st.subheader("Select Runs to Compare")

    run_options = [display for display, _ in all_runs]
    selected_displays = st.multiselect(
        "Choose simulation runs",
        options=run_options[::-1],
        default=run_options[::-1][:2] if len(run_options) >= 2 else run_options[::-1],
        help="Select one or more runs to compare"
    )

    selected_runs = [(display, path) for display, path in all_runs if display in selected_displays]

    # Always include baselines
    comparison_runs = []
    if baseline_8b.exists():
        comparison_runs.append(("Baseline: 8b", baseline_8b))
    if baseline_32b.exists():
        comparison_runs.append(("Baseline: 32b", baseline_32b))

    comparison_runs.extend(selected_runs)

    if len(comparison_runs) < 2:
        st.info("👆 Select at least one run to compare against baselines")
        return

    # Metric selection
    st.subheader("Select Metric")
    metric = st.selectbox(
        "Comparison Metric",
        options=["Prover Calls", "SFLOPs", "Used Lemmas", "Solve Rate"],
        help="Choose which metric to compare across runs"
    )

    metric_map = {
        "Prover Calls": "avg_prover_calls",
        "SFLOPs": "avg_sflops",
        "Used Lemmas": "avg_used_lemmas",
        "Solve Rate": "solve_rate"
    }
    metric_col = metric_map[metric]

    # Load all summaries
    st.markdown("---")
    with st.spinner("Loading data..."):
        summaries = {}
        for display, path in comparison_runs:
            df = load_summary_csv_or_json(path)
            if df is not None:
                summaries[display] = df

    if not summaries:
        st.error("Failed to load any summary data.")
        return

    st.success(f"✅ Loaded {len(summaries)} runs")

    # Create short names for table columns (strategy + config)
    run_short_names = {}
    for display, path in comparison_runs:
        if "Baseline: 8b" in display:
            run_short_names[display] = "Baseline 8b"
        elif "Baseline: 32b" in display:
            run_short_names[display] = "Baseline 32b"
        else:
            # Extract strategy and config
            strategy = get_strategy(path)
            strategy_short = translate_strategy_name(strategy) if strategy else "Unknown"

            # Get config details
            config_parts = []
            if " - " in display:
                config_str = display.split(" - ", 1)[1]
                # Remove strategy if already in config_str
                if strategy:
                    strategy_translated = translate_strategy_name(strategy)
                    if config_str.startswith(strategy_translated):
                        # Extract just the params part
                        parts = config_str.split(" | ", 1)
                        if len(parts) > 1:
                            config_parts = parts[1].split(" | ")
                    else:
                        config_parts = config_str.split(" | ")
                else:
                    config_parts = config_str.split(" | ")

            # Build short descriptive name
            if config_parts:
                # Take first 3 most important config params
                short_config = " | ".join(config_parts[:3])
                run_short_names[display] = f"{strategy_short}: {short_config}"
            else:
                run_short_names[display] = strategy_short

    # Find common problems across ALL runs (inner join)
    common_problems = None
    for display, df in summaries.items():
        problem_set = set(df["problem_id"])
        if common_problems is None:
            common_problems = problem_set
        else:
            common_problems = common_problems.intersection(problem_set)

    if not common_problems:
        st.error("No common problems found across all runs")
        return

    reference_problems = common_problems
    st.info(f"Showing {len(reference_problems)} common problems across all {len(summaries)} runs")

    st.markdown("---")
    st.subheader(f"Comparison Table: {metric}")

    # Build comparison table
    table_data = []

    for problem_id in sorted(reference_problems):
        row = {"Problem": problem_id}
        values = {}

        for run_display in summaries.keys():
            short_name = run_short_names.get(run_display, run_display)
            df = summaries[run_display]
            problem_row = df[df["problem_id"] == problem_id]

            if problem_row.empty:
                row[short_name] = None  # Keep as None for proper sorting
                values[short_name] = None
            else:
                value = problem_row.iloc[0][metric_col]
                # Keep numeric values for sorting
                # For Solve Rate, multiply by 100 for percentage display
                display_value = value * 100 if metric == "Solve Rate" else value
                row[short_name] = display_value if value > 0 else None
                values[short_name] = value if value > 0 else None

        # Find best performer (lowest for calls/sflops, highest for solve rate)
        if metric in ["Solve Rate"]:
            # Higher is better
            valid_values = {k: v for k, v in values.items() if v is not None and v > 0}
            if valid_values:
                best_run = max(valid_values, key=valid_values.get)
                row["best"] = best_run
        else:
            # Lower is better
            valid_values = {k: v for k, v in values.items() if v is not None and v > 0}
            if valid_values:
                best_run = min(valid_values, key=valid_values.get)
                row["best"] = best_run

        table_data.append(row)

    # Create styled DataFrame
    df_display = pd.DataFrame(table_data)

    # Store best values before dropping the column
    best_values = df_display["best"].copy() if "best" in df_display.columns else None

    # Drop the "best" column before displaying
    if "best" in df_display.columns:
        df_display = df_display.drop(columns=["best"])

    # Apply styling: highlight best in green
    def highlight_best(row):
        if best_values is None:
            return ["" for _ in row]

        best_run = best_values.iloc[row.name]
        styles = ["" for _ in row]

        if best_run and best_run in row.index:
            idx = row.index.get_loc(best_run)
            styles[idx] = "background-color: #90EE90"

        return styles

    df_styled = df_display.style.apply(highlight_best, axis=1)

    # Configure column formatting
    column_config = {}
    for col in df_display.columns:
        if col != "Problem":
            if metric in ["Prover Calls", "Used Lemmas"]:
                column_config[col] = st.column_config.NumberColumn(col, format="%.2f")
            elif metric == "SFLOPs":
                column_config[col] = st.column_config.NumberColumn(col, format="%.0f")
            elif metric == "Solve Rate":
                column_config[col] = st.column_config.NumberColumn(col, format="%.1f%%")

    st.dataframe(df_styled, use_container_width=True, height=600, column_config=column_config)

    # Show legend for column names
    with st.expander("📖 Run Details Legend"):
        legend_data = []
        for display, path in comparison_runs:
            short_name = run_short_names[display]
            legend_data.append({
                "Column Name": short_name,
                "Full Path": str(path.relative_to(path.parent.parent.parent.parent.parent)) if not display.startswith("Baseline") else display,
                "Timestamp": display.split(" - ")[0] if " - " in display else display
            })
        legend_df = pd.DataFrame(legend_data)
        st.dataframe(legend_df, use_container_width=True, hide_index=True)

    # Summary Statistics with comparisons
    st.markdown("---")
    st.subheader("Summary Statistics")

    # Count wins against baselines and overall best
    summary_stats = []

    # Get baseline columns if they exist
    baseline_8b_col = None
    baseline_32b_col = None
    for display, path in comparison_runs:
        short_name = run_short_names[display]
        if "Baseline 8b" in short_name:
            baseline_8b_col = short_name
        elif "Baseline 32b" in short_name:
            baseline_32b_col = short_name

    for display, path in comparison_runs:
        short_name = run_short_names[display]
        df_sum = summaries[display]

        # Filter to only common problems
        df_common = df_sum[df_sum["problem_id"].isin(reference_problems)]

        total_problems = len(reference_problems)

        # Handle different CSV formats
        # Baseline CSVs only have avg_prover_calls and avg_sflops (no num_seeds_solved)
        # Experiment CSVs have num_seeds_solved, solve_rate, etc.
        if "num_seeds_solved" in df_common.columns:
            problems_solved = df_common["num_seeds_solved"].gt(0).sum()
        elif "solve_rate" in df_common.columns:
            problems_solved = df_common["solve_rate"].gt(0).sum()
        elif "avg_prover_calls" in df_common.columns:
            # For baselines, count problems where avg_prover_calls > 0
            problems_solved = df_common["avg_prover_calls"].gt(0).sum()
        else:
            problems_solved = 0

        # Count wins
        better_than_8b = 0
        better_than_32b = 0
        times_best = 0

        if best_values is not None:
            for idx, problem_id in enumerate(sorted(reference_problems)):
                best_run = best_values.iloc[idx] if idx < len(best_values) else None

                # Check if this run was the best
                if best_run == short_name:
                    times_best += 1

                # Get this run's value (if metric column exists)
                problem_row = df_sum[df_sum["problem_id"] == problem_id]
                if not problem_row.empty and metric_col in problem_row.columns:
                    this_value = problem_row.iloc[0][metric_col]

                    # Compare to 8b baseline
                    if baseline_8b_col:
                        baseline_8b_display = next((d for d, _ in comparison_runs if run_short_names.get(d) == baseline_8b_col), None)
                        if baseline_8b_display and baseline_8b_display in summaries:
                            baseline_8b_df = summaries[baseline_8b_display]
                            baseline_row = baseline_8b_df[baseline_8b_df["problem_id"] == problem_id]
                            if not baseline_row.empty and metric_col in baseline_row.columns:
                                baseline_value = baseline_row.iloc[0][metric_col]
                                # For calls/sflops, lower is better; for solve_rate, higher is better
                                if metric in ["Solve Rate"]:
                                    if this_value > baseline_value and this_value > 0:
                                        better_than_8b += 1
                                else:
                                    if this_value < baseline_value and this_value > 0 and baseline_value > 0:
                                        better_than_8b += 1

                    # Compare to 32b baseline
                    if baseline_32b_col:
                        baseline_32b_display = next((d for d, _ in comparison_runs if run_short_names.get(d) == baseline_32b_col), None)
                        if baseline_32b_display and baseline_32b_display in summaries:
                            baseline_32b_df = summaries[baseline_32b_display]
                            baseline_row = baseline_32b_df[baseline_32b_df["problem_id"] == problem_id]
                            if not baseline_row.empty and metric_col in baseline_row.columns:
                                baseline_value = baseline_row.iloc[0][metric_col]
                                if metric in ["Solve Rate"]:
                                    if this_value > baseline_value and this_value > 0:
                                        better_than_32b += 1
                                else:
                                    if this_value < baseline_value and this_value > 0 and baseline_value > 0:
                                        better_than_32b += 1

        summary_stats.append({
            "Run": short_name,
            "Problems Solved": f"{problems_solved}/{total_problems}",
            "Average Output SFLOPs": f"{df_common['avg_sflops'].mean():.0f}" if 'avg_sflops' in df_common.columns else "N/A",
            "Average Prover Calls": f"{df_common['avg_prover_calls'].mean():.2f}" if 'avg_prover_calls' in df_common.columns else "N/A",
            "Average Used Lemmas": f"{df_common['avg_used_lemmas'].mean():.2f}" if 'avg_used_lemmas' in df_common.columns else "N/A",
            "Better than 8b": f"{better_than_8b}" if baseline_8b_col else "N/A",
            "Better than 32b": f"{better_than_32b}" if baseline_32b_col else "N/A",
            "Times Best": f"{times_best}",
        })

    summary_df = pd.DataFrame(summary_stats)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_simulation_experiments_viewer(run_dir: Path = None) -> None:
    """
    Render the simulation experiments viewer.

    Can browse all runs with experiments or filter to a specific base run.

    Args:
        run_dir: Optional path to the run directory to filter to
    """
    st.header("🧪 Simulation Experiments")

    # If no run_dir specified, browse all runs with simulations
    if not run_dir:
        st.markdown("### Select Base Run")

        # Find all runs with simulations
        all_runs_with_sims = []

        # Check scratch/results
        results_base = Path("scratch/results")
        if not results_base.exists():
            results_base = Path("../scratch/results")
        if not results_base.exists():
            results_base = Path("../../scratch/results")

        if results_base.exists():
            for run_type_dir in results_base.glob("*"):  # e.g., "combined", "seed_prover_chunk_0"
                if run_type_dir.is_dir():
                    for year_dir in run_type_dir.glob("20*"):
                        if year_dir.is_dir():
                            for month_dir in year_dir.glob("*"):
                                if month_dir.is_dir():
                                    for day_dir in month_dir.glob("*"):
                                        if day_dir.is_dir():
                                            for time_dir in day_dir.glob("*"):
                                                if time_dir.is_dir():
                                                    sim_dir = time_dir / "simulations"
                                                    if sim_dir.exists() and any(sim_dir.glob("*/*/*")):
                                                        run_display = f"results/{run_type_dir.name}/{year_dir.name}/{month_dir.name}/{day_dir.name}/{time_dir.name}"
                                                        all_runs_with_sims.append((run_display, time_dir))

        # Check scratch/dump for additional simulations
        dump_base = Path("scratch/dump")
        if not dump_base.exists():
            dump_base = Path("../scratch/dump")
        if not dump_base.exists():
            dump_base = Path("../../scratch/dump")

        if dump_base.exists():
            # Check combined_dump/simulations directly
            combined_dump = dump_base / "combined_dump"
            if combined_dump.exists():
                sim_dir = combined_dump / "simulations"
                if sim_dir.exists() and any(sim_dir.glob("*/*/*")):
                    # Find all simulation experiments in this directory
                    for year_dir in sim_dir.glob("20*"):
                        if year_dir.is_dir():
                            for month_dir in year_dir.glob("*"):
                                if month_dir.is_dir():
                                    for day_dir in month_dir.glob("*"):
                                        if day_dir.is_dir():
                                            for time_dir in day_dir.glob("*"):
                                                if time_dir.is_dir():
                                                    run_display = f"dump/combined_dump/{year_dir.name}/{month_dir.name}/{day_dir.name}/{time_dir.name}"
                                                    all_runs_with_sims.append((run_display, time_dir))

        if not all_runs_with_sims:
            st.info("No runs with simulation experiments found.")
            st.info(
                "Run `python run_simulation_experiments.py` to generate simulation experiments."
            )
            return

        # Sort by newest first
        all_runs_with_sims.sort(reverse=True)

        selected_run_display = st.selectbox(
            "Select Base Run",
            options=[r[0] for r in all_runs_with_sims],
            help="Choose a base run to view its simulation experiments",
        )

        if selected_run_display:
            run_dir = Path([r[1] for r in all_runs_with_sims if r[0] == selected_run_display][0])

    # Check if simulations exist
    if not run_dir:
        return

    simulations_dir = run_dir / "simulations"

    # If simulations not found, check parent directory (for cases like minified/simulations)
    if not simulations_dir.exists() and run_dir.name == "minified":
        parent_simulations = run_dir.parent / "simulations"
        if parent_simulations.exists():
            simulations_dir = parent_simulations
            run_dir = run_dir.parent
            st.info(f"💡 Found simulations in parent directory: `{run_dir.name}/simulations/`")

    # If still not found, check if we're in combined_dump/minified and look for combined_dump/simulations
    if not simulations_dir.exists():
        # Check if we're in a path like .../combined_dump/minified
        if "combined_dump" in str(run_dir):
            # Find combined_dump in the path
            parts = run_dir.parts
            try:
                combined_dump_idx = parts.index("combined_dump")
                combined_dump_path = Path(*parts[:combined_dump_idx+1])
                alt_simulations = combined_dump_path / "simulations"
                if alt_simulations.exists():
                    simulations_dir = alt_simulations
                    st.info(f"💡 Found simulations at: `combined_dump/simulations/`")
            except (ValueError, IndexError):
                pass

    # Debug info
    with st.expander("🔍 Debug: Experiment Search"):
        st.write("**Run directory:**")
        st.code(str(run_dir.resolve()))
        st.write(f"Exists: {run_dir.exists()}")

        st.write("**Simulations directory:**")
        st.code(str(simulations_dir.resolve()))
        st.write(f"Exists: {simulations_dir.exists()}")

        if simulations_dir.exists():
            year_dirs = list(simulations_dir.glob("20*"))
            st.write(f"Year directories found: {len(year_dirs)}")
            if year_dirs:
                st.write("Year dirs:", [d.name for d in year_dirs[:5]])

    if not simulations_dir.exists():
        st.info("No simulation experiments found in this run.")
        st.info(
            "Run `python run_simulation_experiments.py` to generate simulation experiments."
        )
        return

    # Check if we're already at an experiment level (has config.yaml and summary.csv/json)
    is_experiment_dir = (
        (simulations_dir.parent / "config.yaml").exists() or
        (simulations_dir.parent / "summary.csv").exists() or
        (simulations_dir.parent / "summary.json").exists()
    )

    if is_experiment_dir:
        # We're already at an experiment directory, use it directly
        st.markdown(f"**Experiment:** `{simulations_dir.parent.name}`")
        st.markdown("---")

        # Tab selector for view mode
        tab1, tab2 = st.tabs(["📋 Single Experiment View", "📊 Multi-Run Comparison"])

        with tab2:
            # Multi-run comparison view
            render_multi_run_comparison_tab(simulations_dir.parent.parent)
            return  # Exit early - don't render single experiment view

        with tab1:
            selected_exp_dir = simulations_dir.parent

            # Load experiment results
            with st.spinner("Loading experiment results..."):
                experiment_results = load_experiment_results(selected_exp_dir)

            if not experiment_results:
                st.warning("No results found in this experiment.")
                return

            # Load config
            config_path = selected_exp_dir / "config.yaml"
            config = None
            if config_path.exists():
                import yaml

                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

            # Display config info
            st.markdown("### Experiment Configuration")
            if config:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Strategy", config["strategy"]["type"])
                with col2:
                    st.metric("Seeds per Config", config["simulation"].get("num_seeds", 5))

            st.markdown("---")

            # Compute aggregate stats for all parameter combinations
            ranking_data = []

            for param_str, summary_df in experiment_results.items():
                params = parse_param_string(param_str)
                stats = compute_aggregate_stats(summary_df, config)

                # Build row for table
                row = {
                    "param_string": param_str,
                    "params": params,
                    "stats": stats,
                    "df": summary_df,  # Store for drill-down
                }
                ranking_data.append(row)

            # Sort by problems solved (descending)
            ranking_data.sort(key=lambda x: x["stats"]["problems_solved_count"], reverse=True)

            # Display ranking table
            st.markdown("### Hyperparameter Rankings (sorted by avg problems solved)")

            table_data = []
            for i, row in enumerate(ranking_data, 1):
                params = row["params"]
                stats = row["stats"]

                # Build hyperparameter string
                param_display = " | ".join(
                    [f"{k}={v}" for k, v in sorted(params.items())]
                )

                table_data.append(
                    {
                        "Rank": i,
                        "Hyperparameters": param_display,
                        "Problems Solved": f"{stats['problems_solved']}/{stats['total_problems']}",
                        "Avg SFLOPs": f"{stats['avg_output_sflops']:.0f}",
                        "Avg Prover Calls": f"{stats['avg_prover_calls']:.1f}",
                        "Avg Used Lemmas": f"{stats['avg_used_lemmas']:.2f}",
                    }
                )

            # Display as dataframe
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Drill-down into selected hyperparameter combo
            st.markdown("### View Details")

            selected_rank = st.selectbox(
                "Select hyperparameter combination to view details",
                options=list(range(1, len(ranking_data) + 1)),
                format_func=lambda x: f"Rank {x}: {ranking_data[x-1]['params']}",
                key="rank_selector_1"
            )

            if selected_rank:
                selected_row = ranking_data[selected_rank - 1]
                summary_df = selected_row["df"]
                param_str = selected_row["param_string"]

                st.markdown(f"### Per-Problem Summary for: {selected_row['params']}")

                # Display summary statistics
                stats = selected_row["stats"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Problems Solved", f"{stats['problems_solved']}/{stats['total_problems']}")
                with col2:
                    st.metric("Num Seeds", stats['num_seeds'])
                with col3:
                    st.metric("Avg SFLOPs", f"{stats['avg_output_sflops']:.0f}")
                with col4:
                    st.metric("Avg Prover Calls", f"{stats['avg_prover_calls']:.1f}")

                st.markdown("**Per-Problem Results:**")

                # Format the summary dataframe for display
                display_df = summary_df.copy()
                display_df["Solved"] = display_df["num_seeds_solved"].apply(
                    lambda x: f"✅ {x}/{stats['num_seeds']}" if x > 0 else f"❌ 0/{stats['num_seeds']}"
                )
                # Keep numeric columns for sorting, rename for clarity
                display_df["Solve Rate"] = display_df["solve_rate"] * 100  # Convert to percentage
                display_df["Avg Prover Calls"] = display_df["avg_prover_calls"]
                display_df["Avg SFLOPs"] = display_df["avg_sflops"]

                if "avg_used_lemmas" in display_df.columns:
                    display_df["Avg Used Lemmas"] = display_df["avg_used_lemmas"]

                # Select columns to display
                cols_to_show = ["problem_id", "Solved", "Solve Rate", "Avg Prover Calls", "Avg SFLOPs"]
                if "Avg Used Lemmas" in display_df.columns:
                    cols_to_show.append("Avg Used Lemmas")

                # Configure column formatting
                column_config = {
                    "Solve Rate": st.column_config.NumberColumn("Solve Rate", format="%.1f%%"),
                    "Avg Prover Calls": st.column_config.NumberColumn("Avg Prover Calls", format="%.2f"),
                    "Avg SFLOPs": st.column_config.NumberColumn("Avg SFLOPs", format="%.0f"),
                }
                if "Avg Used Lemmas" in cols_to_show:
                    column_config["Avg Used Lemmas"] = st.column_config.NumberColumn("Avg Used Lemmas", format="%.2f")

                st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True, column_config=column_config)

                # Add visualization for ProofLengthPredictor strategy
                if config and config.get("strategy", {}).get("type") == "ProofLengthPredictor":
                    st.markdown("---")
                    st.markdown("### 📈 Proof Length Prediction Visualization")
                    st.markdown("Visualize how the proof length model makes predictions over time for a specific problem.")

                    # Load fine-grained attempts
                    fine_grained_dir = selected_exp_dir / "fine_grained_attempts" if (selected_exp_dir / "fine_grained_attempts").exists() else selected_exp_dir / param_str / "fine_grained_attempts" if (selected_exp_dir / param_str / "fine_grained_attempts").exists() else None

                    if fine_grained_dir and fine_grained_dir.exists():
                        # Get list of available problems
                        problem_files = sorted(fine_grained_dir.glob("*.json"))
                        problem_ids = [f.stem for f in problem_files]

                        if problem_ids:
                            selected_problem = st.selectbox(
                                "Select Problem",
                                options=problem_ids,
                                key=f"plp_problem_1_{selected_rank}"
                            )

                            if selected_problem:
                                attempts_file = fine_grained_dir / f"{selected_problem}.json"

                                if attempts_file.exists():
                                    with open(attempts_file, 'r') as f:
                                        attempts_data = json.load(f)

                                    # Get seed from metadata
                                    seed_num = attempts_data.get("metadata", {}).get("seed", "unknown")
                                    st.info(f"Showing data for seed: **{seed_num}**")

                                    proof_attempts = attempts_data.get("proof_attempts", [])

                                    # Get unique breakdown IDs
                                    breakdown_ids = sorted(set(attempt.get("breakdown_id", -1) for attempt in proof_attempts))

                                    if breakdown_ids:
                                        selected_breakdown = st.selectbox(
                                            "Select Breakdown",
                                            options=breakdown_ids,
                                            format_func=lambda x: f"Breakdown {x}",
                                            key=f"breakdown_selector_{selected_rank}"
                                        )

                                        # Filter attempts to selected breakdown
                                        proof_attempts = [a for a in proof_attempts if a.get("breakdown_id") == selected_breakdown]
                                    else:
                                        st.warning("No breakdown IDs found in attempts data")

                                    # Group attempts by target (theorem/lemma)
                                    attempts_by_target = {}

                                    for attempt in proof_attempts:
                                        params_dict = attempt.get("strategy_params_at_decision", {})

                                        if params_dict.get("strategy") == "proof_length_predictor":
                                            attempt_num = attempt.get("attempt_number", 0)
                                            lemma_id = attempt.get("lemma_id", -999)

                                            # Create target key
                                            if lemma_id == -1:
                                                target_key = "Theorem"
                                            elif lemma_id >= 0:
                                                target_key = f"Lemma {lemma_id}"
                                            else:
                                                target_key = f"Unknown (lemma_id={lemma_id})"

                                            if target_key not in attempts_by_target:
                                                attempts_by_target[target_key] = {
                                                    "attempt_numbers": [],
                                                    "predictions": [],
                                                    "thresholds": [],
                                                    "avg_proof_lengths": [],
                                                    "avg_num_errors_list": [],
                                                    "modes": [],
                                                    "is_passing": [],
                                                    "is_complete": []
                                                }

                                            prediction = params_dict.get("last_prediction")
                                            threshold = params_dict.get("current_threshold")
                                            avg_proof_len = params_dict.get("avg_proof_length")
                                            avg_errors = params_dict.get("avg_num_errors")
                                            mode = params_dict.get("mode", "8b")
                                            is_passing = attempt.get("is_passing", False)
                                            is_complete = attempt.get("is_complete", False)

                                            attempts_by_target[target_key]["attempt_numbers"].append(attempt_num)
                                            attempts_by_target[target_key]["predictions"].append(prediction)
                                            attempts_by_target[target_key]["thresholds"].append(threshold)
                                            attempts_by_target[target_key]["avg_proof_lengths"].append(avg_proof_len)
                                            attempts_by_target[target_key]["avg_num_errors_list"].append(avg_errors)
                                            attempts_by_target[target_key]["modes"].append(mode)
                                            attempts_by_target[target_key]["is_passing"].append(is_passing)
                                            attempts_by_target[target_key]["is_complete"].append(is_complete)

                                    if attempts_by_target:
                                        # Sort targets: Theorem first, then Lemmas by ID
                                        def sort_key(target):
                                            if target == "Theorem":
                                                return (0, 0)
                                            elif target.startswith("Lemma "):
                                                try:
                                                    lemma_id = int(target.split(" ")[1])
                                                    return (1, lemma_id)
                                                except:
                                                    return (2, target)
                                            else:
                                                return (2, target)

                                        sorted_targets = sorted(attempts_by_target.keys(), key=sort_key)

                                        # Create a plot for each target
                                        for target_key in sorted_targets:
                                            data = attempts_by_target[target_key]
                                            attempt_numbers = data["attempt_numbers"]
                                            predictions = data["predictions"]
                                            thresholds = data["thresholds"]
                                            avg_proof_lengths = data["avg_proof_lengths"]
                                            avg_num_errors_list = data["avg_num_errors_list"]
                                            modes = data["modes"]
                                            is_passing_list = data["is_passing"]
                                            is_complete_list = data["is_complete"]

                                            st.markdown(f"#### {target_key}")

                                            # Create figure
                                            fig = go.Figure()

                                            # Add prediction line
                                            valid_predictions = [(i, p) for i, p in zip(attempt_numbers, predictions) if p is not None]
                                            if valid_predictions:
                                                pred_attempts, pred_values = zip(*valid_predictions)
                                                fig.add_trace(go.Scatter(
                                                    x=pred_attempts,
                                                    y=pred_values,
                                                    mode='lines+markers',
                                                    name='Predicted Proof Length',
                                                    line=dict(color='blue', width=2),
                                                    marker=dict(size=8)
                                                ))

                                            # Add threshold line(s)
                                            if thresholds:
                                                # Get unique thresholds and their ranges
                                                threshold_changes = []
                                                for i, t in enumerate(thresholds):
                                                    if i == 0 or t != thresholds[i-1]:
                                                        threshold_changes.append((attempt_numbers[i], t))

                                                # Draw threshold segments
                                                for idx, (start_attempt, threshold_val) in enumerate(threshold_changes):
                                                    if threshold_val is not None:
                                                        end_attempt = threshold_changes[idx+1][0] if idx+1 < len(threshold_changes) else attempt_numbers[-1]

                                                        fig.add_trace(go.Scatter(
                                                            x=[start_attempt, end_attempt],
                                                            y=[threshold_val, threshold_val],
                                                            mode='lines',
                                                            name=f'Threshold = {threshold_val:.0f}',
                                                            line=dict(color='red', width=2, dash='dot'),
                                                            showlegend=(idx == 0 or threshold_val != threshold_changes[idx-1][1])
                                                        ))

                                            # Add markers for model upgrades
                                            upgrade_attempts = [i for i, m in zip(attempt_numbers, modes) if m == "32b"]
                                            if upgrade_attempts:
                                                upgrade_predictions = [predictions[attempt_numbers.index(a)] for a in upgrade_attempts if predictions[attempt_numbers.index(a)] is not None]
                                                if upgrade_predictions:
                                                    fig.add_trace(go.Scatter(
                                                        x=upgrade_attempts[:len(upgrade_predictions)],
                                                        y=upgrade_predictions,
                                                        mode='markers',
                                                        name='Upgraded to 32b',
                                                        marker=dict(size=12, color='orange', symbol='star')
                                                    ))

                                            fig.update_layout(
                                                title=f"{target_key} - Proof Length Predictions (seed={seed_num})",
                                                xaxis_title="Attempt Number",
                                                yaxis_title="Predicted Proof Length",
                                                hovermode='x unified',
                                                height=400
                                            )

                                            st.plotly_chart(fig, use_container_width=True)

                                            # Show additional metrics in expandable section
                                            with st.expander(f"📊 Detailed Metrics for {target_key}"):
                                                metrics_data = []
                                                for i, attempt_num in enumerate(attempt_numbers):
                                                    # Determine solved status
                                                    is_passing = is_passing_list[i]
                                                    is_complete = is_complete_list[i]
                                                    if is_passing and is_complete:
                                                        solved_status = "✅ Solved"
                                                    elif is_passing:
                                                        solved_status = "⚠️ Pass (incomplete)"
                                                    else:
                                                        solved_status = "❌ Failed"

                                                    metrics_data.append({
                                                        "Attempt": attempt_num,
                                                        "Solved": solved_status,
                                                        "Prediction": f"{predictions[i]:.2f}" if predictions[i] is not None else "N/A",
                                                        "Threshold": f"{thresholds[i]:.0f}" if thresholds[i] is not None else "N/A",
                                                        "Avg Proof Length": f"{avg_proof_lengths[i]:.1f}" if avg_proof_lengths[i] is not None else "N/A",
                                                        "Avg Num Errors": f"{avg_num_errors_list[i]:.1f}" if avg_num_errors_list[i] is not None else "N/A",
                                                        "Mode": modes[i]
                                                    })

                                                metrics_df = pd.DataFrame(metrics_data)
                                                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No ProofLengthPredictor data found for this problem.")
                                else:
                                    st.warning(f"No fine-grained attempts file found for: {selected_problem}")
                        else:
                            st.info("No problems with fine-grained data found.")
                    else:
                        st.info("Fine-grained attempts directory not found for this experiment.")
    else:
        # Get available experiments
        experiments = get_simulation_runs(run_dir)

        if not experiments:
            st.warning("No simulation experiments found in this run.")
            st.info("The simulations directory exists but contains no valid experiment timestamps.")
            return

        st.markdown(f"**Base Run:** `{run_dir.name}`")
        st.markdown("---")

        # Tab selector for view mode
        tab1, tab2 = st.tabs(["📋 Single Experiment View", "📊 Multi-Run Comparison"])

        with tab2:
            # Multi-run comparison view
            render_multi_run_comparison_tab(run_dir)

        with tab1:
            # Experiment selector
            experiment_options = [
            f"{exp.parent.parent.parent.name}/{exp.parent.parent.name}/{exp.parent.name}/{exp.name}"
            for exp in experiments
            ]
            selected_exp = st.selectbox(
            "Select Experiment",
            options=experiment_options,
            help="Choose a simulation experiment to view results",
            key="exp_selector",
            )

            if not selected_exp:
                return

            selected_exp_dir = experiments[experiment_options.index(selected_exp)]

            # Load experiment results
            with st.spinner("Loading experiment results..."):
                experiment_results = load_experiment_results(selected_exp_dir)

            if not experiment_results:
                st.warning("No results found in this experiment.")
                return

            # Load config
            config_path = selected_exp_dir / "config.yaml"
            config = None
            if config_path.exists():
                import yaml

                with open(config_path, "r") as f:
                    config = yaml.safe_load(f)

            # Display config info
            st.markdown("### Experiment Configuration")
            if config:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Strategy", config["strategy"]["type"])
                with col2:
                    st.metric("Seeds per Config", config["simulation"].get("num_seeds", 5))

            st.markdown("---")

            # Compute aggregate stats for all parameter combinations
            ranking_data = []

            for param_str, summary_df in experiment_results.items():
                params = parse_param_string(param_str)
                stats = compute_aggregate_stats(summary_df, config)

                # Build row for table
                row = {
                    "param_string": param_str,
                    "params": params,
                    "stats": stats,
                    "df": summary_df,  # Store for drill-down
                }
                ranking_data.append(row)

            # Sort by problems solved (descending)
            ranking_data.sort(key=lambda x: x["stats"]["problems_solved_count"], reverse=True)

            # Display ranking table
            st.markdown("### Hyperparameter Rankings (sorted by avg problems solved)")

            table_data = []
            for i, row in enumerate(ranking_data, 1):
                params = row["params"]
                stats = row["stats"]

                # Build hyperparameter string
                param_display = " | ".join(
                    [f"{k}={v}" for k, v in sorted(params.items())]
                )

                table_data.append(
                    {
                        "Rank": i,
                        "Hyperparameters": param_display,
                        "Problems Solved": f"{stats['problems_solved']}/{stats['total_problems']}",
                        "Avg SFLOPs": f"{stats['avg_output_sflops']:.0f}",
                        "Avg Prover Calls": f"{stats['avg_prover_calls']:.1f}",
                        "Avg Used Lemmas": f"{stats['avg_used_lemmas']:.2f}",
                    }
                )

            # Display as dataframe
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("---")

            # Drill-down into selected hyperparameter combo
            st.markdown("### View Details")

            selected_rank = st.selectbox(
                "Select hyperparameter combination to view details",
                options=list(range(1, len(ranking_data) + 1)),
                format_func=lambda x: f"Rank {x}: {ranking_data[x-1]['params']}",
            )

            if selected_rank:
                selected_row = ranking_data[selected_rank - 1]
                summary_df = selected_row["df"]

                st.markdown(f"### Per-Problem Summary for: {selected_row['params']}")

                # Display summary statistics
                stats = selected_row["stats"]
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Problems Solved", f"{stats['problems_solved']}/{stats['total_problems']}")
                with col2:
                    st.metric("Num Seeds", stats['num_seeds'])
                with col3:
                    st.metric("Avg SFLOPs", f"{stats['avg_output_sflops']:.0f}")
                with col4:
                    st.metric("Avg Prover Calls", f"{stats['avg_prover_calls']:.1f}")

                st.markdown("**Per-Problem Results:**")

                # Format the summary dataframe for display
                display_df = summary_df.copy()
                display_df["Solved"] = display_df["num_seeds_solved"].apply(
                    lambda x: f"✅ {x}/{stats['num_seeds']}" if x > 0 else f"❌ 0/{stats['num_seeds']}"
                )
                # Keep numeric columns for sorting, rename for clarity
                display_df["Solve Rate"] = display_df["solve_rate"] * 100  # Convert to percentage
                display_df["Avg Prover Calls"] = display_df["avg_prover_calls"]
                display_df["Avg SFLOPs"] = display_df["avg_sflops"]

                if "avg_used_lemmas" in display_df.columns:
                    display_df["Avg Used Lemmas"] = display_df["avg_used_lemmas"]

                # Select columns to display
                cols_to_show = ["problem_id", "Solved", "Solve Rate", "Avg Prover Calls", "Avg SFLOPs"]
                if "Avg Used Lemmas" in display_df.columns:
                    cols_to_show.append("Avg Used Lemmas")

                # Configure column formatting
                column_config = {
                    "Solve Rate": st.column_config.NumberColumn("Solve Rate", format="%.1f%%"),
                    "Avg Prover Calls": st.column_config.NumberColumn("Avg Prover Calls", format="%.2f"),
                    "Avg SFLOPs": st.column_config.NumberColumn("Avg SFLOPs", format="%.0f"),
                }
                if "Avg Used Lemmas" in cols_to_show:
                    column_config["Avg Used Lemmas"] = st.column_config.NumberColumn("Avg Used Lemmas", format="%.2f")

                st.dataframe(display_df[cols_to_show], use_container_width=True, hide_index=True, column_config=column_config)

                # Add visualization for ProofLengthPredictor strategy
                if config and config.get("strategy", {}).get("type") == "ProofLengthPredictor":
                    st.markdown("---")
                    st.markdown("### 📈 Proof Length Prediction Visualization")
                    st.markdown("Visualize how the proof length model makes predictions over time for a specific problem.")

                    # Load fine-grained attempts
                    fine_grained_dir = selected_exp_dir / "fine_grained_attempts" if (selected_exp_dir / "fine_grained_attempts").exists() else selected_exp_dir / param_str / "fine_grained_attempts" if (selected_exp_dir / param_str / "fine_grained_attempts").exists() else None

                    if fine_grained_dir and fine_grained_dir.exists():
                        # Get list of available problems
                        problem_files = sorted(fine_grained_dir.glob("*.json"))
                        problem_ids = [f.stem for f in problem_files]

                        if problem_ids:
                            selected_problem = st.selectbox(
                                "Select Problem",
                                options=problem_ids,
                                key=f"plp_problem_{selected_rank}"
                            )

                            if selected_problem:
                                attempts_file = fine_grained_dir / f"{selected_problem}.json"

                                if attempts_file.exists():
                                    with open(attempts_file, 'r') as f:
                                        attempts_data = json.load(f)

                                    # Get seed from metadata
                                    seed_num = attempts_data.get("metadata", {}).get("seed", "unknown")
                                    st.info(f"Showing data for seed: **{seed_num}**")

                                    proof_attempts = attempts_data.get("proof_attempts", [])

                                    # Get unique breakdown IDs
                                    breakdown_ids = sorted(set(attempt.get("breakdown_id", -1) for attempt in proof_attempts))

                                    if breakdown_ids:
                                        selected_breakdown = st.selectbox(
                                            "Select Breakdown",
                                            options=breakdown_ids,
                                            format_func=lambda x: f"Breakdown {x}",
                                            key=f"breakdown_selector_{selected_rank}"
                                        )

                                        # Filter attempts to selected breakdown
                                        proof_attempts = [a for a in proof_attempts if a.get("breakdown_id") == selected_breakdown]
                                    else:
                                        st.warning("No breakdown IDs found in attempts data")

                                    # Group attempts by target (theorem/lemma)
                                    attempts_by_target = {}

                                    for attempt in proof_attempts:
                                        params_dict = attempt.get("strategy_params_at_decision", {})

                                        if params_dict.get("strategy") == "proof_length_predictor":
                                            attempt_num = attempt.get("attempt_number", 0)
                                            lemma_id = attempt.get("lemma_id", -999)

                                            # Create target key
                                            if lemma_id == -1:
                                                target_key = "Theorem"
                                            elif lemma_id >= 0:
                                                target_key = f"Lemma {lemma_id}"
                                            else:
                                                target_key = f"Unknown (lemma_id={lemma_id})"

                                            if target_key not in attempts_by_target:
                                                attempts_by_target[target_key] = {
                                                    "attempt_numbers": [],
                                                    "predictions": [],
                                                    "thresholds": [],
                                                    "avg_proof_lengths": [],
                                                    "avg_num_errors_list": [],
                                                    "modes": [],
                                                    "is_passing": [],
                                                    "is_complete": []
                                                }

                                            prediction = params_dict.get("last_prediction")
                                            threshold = params_dict.get("current_threshold")
                                            avg_proof_len = params_dict.get("avg_proof_length")
                                            avg_errors = params_dict.get("avg_num_errors")
                                            mode = params_dict.get("mode", "8b")
                                            is_passing = attempt.get("is_passing", False)
                                            is_complete = attempt.get("is_complete", False)

                                            attempts_by_target[target_key]["attempt_numbers"].append(attempt_num)
                                            attempts_by_target[target_key]["predictions"].append(prediction)
                                            attempts_by_target[target_key]["thresholds"].append(threshold)
                                            attempts_by_target[target_key]["avg_proof_lengths"].append(avg_proof_len)
                                            attempts_by_target[target_key]["avg_num_errors_list"].append(avg_errors)
                                            attempts_by_target[target_key]["modes"].append(mode)
                                            attempts_by_target[target_key]["is_passing"].append(is_passing)
                                            attempts_by_target[target_key]["is_complete"].append(is_complete)

                                    if attempts_by_target:
                                        # Sort targets: Theorem first, then Lemmas by ID
                                        def sort_key(target):
                                            if target == "Theorem":
                                                return (0, 0)
                                            elif target.startswith("Lemma "):
                                                try:
                                                    lemma_id = int(target.split(" ")[1])
                                                    return (1, lemma_id)
                                                except:
                                                    return (2, target)
                                            else:
                                                return (2, target)

                                        sorted_targets = sorted(attempts_by_target.keys(), key=sort_key)

                                        # Create a plot for each target
                                        for target_key in sorted_targets:
                                            data = attempts_by_target[target_key]
                                            attempt_numbers = data["attempt_numbers"]
                                            predictions = data["predictions"]
                                            thresholds = data["thresholds"]
                                            avg_proof_lengths = data["avg_proof_lengths"]
                                            avg_num_errors_list = data["avg_num_errors_list"]
                                            modes = data["modes"]
                                            is_passing_list = data["is_passing"]
                                            is_complete_list = data["is_complete"]

                                            st.markdown(f"#### {target_key}")

                                            # Create figure
                                            fig = go.Figure()

                                            # Add prediction line
                                            valid_predictions = [(i, p) for i, p in zip(attempt_numbers, predictions) if p is not None]
                                            if valid_predictions:
                                                pred_attempts, pred_values = zip(*valid_predictions)
                                                fig.add_trace(go.Scatter(
                                                    x=pred_attempts,
                                                    y=pred_values,
                                                    mode='lines+markers',
                                                    name='Predicted Proof Length',
                                                    line=dict(color='blue', width=2),
                                                    marker=dict(size=8)
                                                ))

                                            # Add threshold line(s)
                                            if thresholds:
                                                # Get unique thresholds and their ranges
                                                threshold_changes = []
                                                for i, t in enumerate(thresholds):
                                                    if i == 0 or t != thresholds[i-1]:
                                                        threshold_changes.append((attempt_numbers[i], t))

                                                # Draw threshold segments
                                                for idx, (start_attempt, threshold_val) in enumerate(threshold_changes):
                                                    if threshold_val is not None:
                                                        end_attempt = threshold_changes[idx+1][0] if idx+1 < len(threshold_changes) else attempt_numbers[-1]

                                                        fig.add_trace(go.Scatter(
                                                            x=[start_attempt, end_attempt],
                                                            y=[threshold_val, threshold_val],
                                                            mode='lines',
                                                            name=f'Threshold = {threshold_val:.0f}',
                                                            line=dict(color='red', width=2, dash='dot'),
                                                            showlegend=(idx == 0 or threshold_val != threshold_changes[idx-1][1])
                                                        ))

                                            # Add markers for model upgrades
                                            upgrade_attempts = [i for i, m in zip(attempt_numbers, modes) if m == "32b"]
                                            if upgrade_attempts:
                                                upgrade_predictions = [predictions[attempt_numbers.index(a)] for a in upgrade_attempts if predictions[attempt_numbers.index(a)] is not None]
                                                if upgrade_predictions:
                                                    fig.add_trace(go.Scatter(
                                                        x=upgrade_attempts[:len(upgrade_predictions)],
                                                        y=upgrade_predictions,
                                                        mode='markers',
                                                        name='Upgraded to 32b',
                                                        marker=dict(size=12, color='orange', symbol='star')
                                                    ))

                                            fig.update_layout(
                                                title=f"{target_key} - Proof Length Predictions (seed={seed_num})",
                                                xaxis_title="Attempt Number",
                                                yaxis_title="Predicted Proof Length",
                                                hovermode='x unified',
                                                height=400
                                            )

                                            st.plotly_chart(fig, use_container_width=True)

                                            # Show additional metrics in expandable section
                                            with st.expander(f"📊 Detailed Metrics for {target_key}"):
                                                metrics_data = []
                                                for i, attempt_num in enumerate(attempt_numbers):
                                                    # Determine solved status
                                                    is_passing = is_passing_list[i]
                                                    is_complete = is_complete_list[i]
                                                    if is_passing and is_complete:
                                                        solved_status = "✅ Solved"
                                                    elif is_passing:
                                                        solved_status = "⚠️ Pass (incomplete)"
                                                    else:
                                                        solved_status = "❌ Failed"

                                                    metrics_data.append({
                                                        "Attempt": attempt_num,
                                                        "Solved": solved_status,
                                                        "Prediction": f"{predictions[i]:.2f}" if predictions[i] is not None else "N/A",
                                                        "Threshold": f"{thresholds[i]:.0f}" if thresholds[i] is not None else "N/A",
                                                        "Avg Proof Length": f"{avg_proof_lengths[i]:.1f}" if avg_proof_lengths[i] is not None else "N/A",
                                                        "Avg Num Errors": f"{avg_num_errors_list[i]:.1f}" if avg_num_errors_list[i] is not None else "N/A",
                                                        "Mode": modes[i]
                                                    })

                                                metrics_df = pd.DataFrame(metrics_data)
                                                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No ProofLengthPredictor data found for this problem.")
                                else:
                                    st.warning(f"No fine-grained attempts file found for: {selected_problem}")
                        else:
                            st.info("No problems with fine-grained data found.")
                    else:
                        st.info("Fine-grained attempts directory not found for this experiment.")
