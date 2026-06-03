"""
Simulation Comparison Viewer

Compare summary results across multiple simulation runs.
Shows a table with problems as rows and runs as columns.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# Baseline runs paths (relative, will be resolved at runtime)
BASELINE_8B_REL = "full_proof_8b/2025/12/13/simulations_1024"
BASELINE_32B_REL = "full_proof_32b/2025/12/13/simulations_1024"


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
        "initial_threshold": "it",
        "initial_attempts": "ia",
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


def load_summary_data(run_dir: Path) -> Optional[Dict]:
    """
    Load summary data from either summary.json or summary.csv.

    Args:
        run_dir: Path to simulation run

    Returns:
        Dict mapping problem_id -> stats, or None if file not found
    """
    # Try JSON first
    json_path = run_dir / "summary.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Normalize format: ensure all entries have consistent keys
                for problem_id, stats in data.items():
                    if "avg_attempts" in stats and "avg_prover_calls" not in stats:
                        stats["avg_prover_calls"] = stats["avg_attempts"]
                return data
        except Exception as e:
            st.error(f"Error loading {json_path}: {e}")
            return None

    # Try CSV
    csv_path = run_dir / "summary.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)

            # Infer total_seeds from solve_rate and num_seeds_solved
            # solve_rate = num_seeds_solved / total_seeds
            # So: total_seeds = num_seeds_solved / solve_rate (when solve_rate > 0)
            total_seeds_inferred = None
            for _, row in df.iterrows():
                if row.get("solve_rate", 0) > 0:
                    total_seeds_inferred = int(row["num_seeds_solved"] / row["solve_rate"])
                    break

            # Fallback to 16 if we couldn't infer
            if total_seeds_inferred is None:
                total_seeds_inferred = 16

            # Convert to dict format matching JSON structure
            result = {}
            for _, row in df.iterrows():
                problem_id = row["problem_id"]
                result[problem_id] = {
                    "seeds_solved": int(row.get("num_seeds_solved", 0)),
                    "total_seeds": total_seeds_inferred,
                    "avg_prover_calls": float(row.get("avg_prover_calls", 0)),
                    "avg_sflops": float(row.get("avg_sflops", 0)),
                    "avg_used_lemmas": float(row.get("avg_used_lemmas", 0.0)) if "avg_used_lemmas" in row else None,
                    "solved": row.get("num_seeds_solved", 0) > 0
                }
            return result
        except Exception as e:
            st.error(f"Error loading {csv_path}: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None

    return None


def find_simulation_runs(results_base: Path, dump_base: Optional[Path] = None) -> List[Tuple[str, Path]]:
    """
    Find all simulation runs with summary.json or summary.csv files.

    Args:
        results_base: Base results directory (e.g., scratch/results)
        dump_base: Optional dump directory (e.g., scratch/dump)

    Returns:
        List of (display_name, path) tuples
    """
    simulation_runs = []

    # Helper to check if directory has summary file
    def has_summary(path: Path) -> bool:
        return (path / "summary.json").exists() or (path / "summary.csv").exists()

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

    # Search results directory
    if results_base.exists():
        # Pattern: results/{run_type}/{year}/{month}/{day}/{time}/simulations*
        for run_type_dir in results_base.glob("*"):
            if not run_type_dir.is_dir():
                continue

            for year_dir in run_type_dir.glob("20*"):
                if not year_dir.is_dir():
                    continue

                for month_dir in year_dir.glob("*"):
                    if not month_dir.is_dir():
                        continue

                    for day_dir in month_dir.glob("*"):
                        if not day_dir.is_dir():
                            continue

                        for time_dir in day_dir.glob("*"):
                            if not time_dir.is_dir():
                                continue

                            # Check for simulations* directories
                            for sim_dir in time_dir.glob("simulations*"):
                                if sim_dir.is_dir() and has_summary(sim_dir):
                                    timestamp = f"{year_dir.name}/{month_dir.name}/{day_dir.name} {time_dir.name}"
                                    strategy = get_strategy(sim_dir)
                                    strategy_short = translate_strategy_name(strategy) if strategy else sim_dir.name
                                    display = f"{timestamp} - {strategy_short}"
                                    simulation_runs.append((display, sim_dir))

    # Search dump directory
    if dump_base and dump_base.exists():
        # Pattern: dump/{dump_name}/simulations/{year}/{month}/{day}/{time}
        for dump_dir in dump_base.glob("*"):
            if not dump_dir.is_dir():
                continue

            simulations_dir = dump_dir / "simulations"
            if not simulations_dir.exists():
                continue

            for year_dir in simulations_dir.glob("20*"):
                if not year_dir.is_dir():
                    continue

                for month_dir in year_dir.glob("*"):
                    if not month_dir.is_dir():
                        continue

                    for day_dir in month_dir.glob("*"):
                        if not day_dir.is_dir():
                            continue

                        for time_dir in day_dir.glob("*"):
                            if not time_dir.is_dir():
                                continue

                            # Check if summary is directly in time_dir
                            if has_summary(time_dir):
                                timestamp = f"{year_dir.name}/{month_dir.name}/{day_dir.name} {time_dir.name}"
                                strategy = get_strategy(time_dir)
                                strategy_short = translate_strategy_name(strategy) if strategy else ""
                                display = f"{timestamp} - {strategy_short}" if strategy_short else timestamp
                                simulation_runs.append((display, time_dir))
                            else:
                                # Check for hyperparameter subdirectories (e.g., "auto_fit=True_...")
                                for param_dir in time_dir.iterdir():
                                    if param_dir.is_dir() and has_summary(param_dir):
                                        timestamp = f"{year_dir.name}/{month_dir.name}/{day_dir.name} {time_dir.name}"
                                        strategy = get_strategy(param_dir)
                                        config_pretty = prettify_config(param_dir.name, strategy)
                                        display = f"{timestamp} - {config_pretty}"
                                        simulation_runs.append((display, param_dir))

    return sorted(simulation_runs, reverse=True)


def render_simulation_comparison_viewer():
    """
    Main render function for simulation comparison viewer.
    """
    st.header("🔬 Simulation Comparison")

    st.markdown("""
    Compare simulation results across multiple runs. Select runs to compare,
    and the table will show per-problem metrics (seeds solved, prover calls, SFLOPs).

    **Baselines** (8b and 32b full_proof) are always included for reference.
    """)

    st.markdown("---")

    # Find available simulation runs
    # Try different relative paths (from GUI directory structure)
    results_base = None
    for path in [Path("scratch/results"), Path("../scratch/results"), Path("../../scratch/results")]:
        if path.exists():
            results_base = path
            break

    if not results_base:
        results_base = Path("scratch/results")  # Default fallback

    dump_base = None
    for path in [Path("scratch/dump"), Path("../scratch/dump"), Path("../../scratch/dump")]:
        if path.exists():
            dump_base = path
            break

    available_runs = find_simulation_runs(results_base, dump_base)

    # Show debug info in expander
    with st.expander("🔍 Debug: Search Paths"):
        st.write("**Results directory:**")
        st.code(f"{results_base.resolve()}")
        st.write(f"Exists: {results_base.exists()}")

        if dump_base:
            st.write("**Dump directory:**")
            st.code(f"{dump_base.resolve()}")
            st.write(f"Exists: {dump_base.exists()}")
        else:
            st.write("**Dump directory:** Not found")

        st.write(f"**Total simulation runs found:** {len(available_runs)}")

    if not available_runs:
        st.warning("No simulation runs found with summary files.")
        st.info("Run simulations to generate summary.json or summary.csv files for comparison.")
        return

    # Multi-select for comparison runs
    st.subheader("Select Runs to Compare")

    selected_displays = st.multiselect(
        "Choose simulation runs",
        options=[display for display, _ in available_runs],
        default=[available_runs[0][0]] if available_runs else [],
        help="Select one or more simulation runs to compare"
    )

    # Get paths for selected runs
    selected_runs = [(display, path) for display, path in available_runs if display in selected_displays]

    # Always add baselines (if they exist)
    all_runs = []
    if results_base:
        baseline_8b = results_base / BASELINE_8B_REL
        baseline_32b = results_base / BASELINE_32B_REL

        if baseline_8b.exists() and (baseline_8b / "summary.json").exists():
            all_runs.append(("Baseline: 8b (1024 seeds)", baseline_8b))
        else:
            st.info(f"Baseline 8b not found at: {baseline_8b}")

        if baseline_32b.exists() and (baseline_32b / "summary.json").exists():
            all_runs.append(("Baseline: 32b (1024 seeds)", baseline_32b))
        else:
            st.info(f"Baseline 32b not found at: {baseline_32b}")

    all_runs.extend(selected_runs)

    if not selected_runs:
        st.info("👆 Select simulation runs above to compare" + (" against baselines" if len(all_runs) > 0 else ""))
        return

    if not all_runs:
        st.error("No runs to compare. Please check that baseline paths are correct.")
        return

    st.markdown("---")

    # Load all summary data
    st.subheader("Loading Data...")
    progress = st.progress(0)

    summaries = {}
    for i, (display, path) in enumerate(all_runs):
        summary = load_summary_data(path)
        if summary:
            summaries[display] = summary
        else:
            st.warning(f"Could not load summary for: {display}")
        progress.progress((i + 1) / len(all_runs))

    progress.empty()

    if not summaries:
        st.error("No summary data loaded. Check that summary.json files exist.")
        return

    st.success(f"✅ Loaded {len(summaries)} runs")

    # Determine which problems to show (from first selected run, or first baseline if none selected)
    reference_run = selected_runs[0][0] if selected_runs else "Baseline: 8b (1024 seeds)"
    reference_problems = set(summaries[reference_run].keys())

    st.info(f"Showing {len(reference_problems)} problems from reference run: **{reference_run}**")

    st.markdown("---")

    # Build comparison table
    st.subheader("Comparison Table")

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

    # Create shortened display names for columns (strategy + config)
    run_short_names = {}
    for i, (run_display, run_path) in enumerate(all_runs):
        if "Baseline: 8b" in run_display:
            run_short_names[run_display] = "Baseline 8b"
        elif "Baseline: 32b" in run_display:
            run_short_names[run_display] = "Baseline 32b"
        else:
            # Extract strategy and config
            strategy = get_strategy(run_path)
            strategy_short = translate_strategy_name(strategy) if strategy else "Unknown"

            # Extract meaningful parts from path
            parts = run_display.split("/")

            # For hyperparameter runs with long parameter strings
            if len(parts) > 0 and "=" in parts[-1]:
                param_str = parts[-1]
                config_pretty = prettify_config(param_str, strategy)

                # Remove strategy prefix if already included
                if strategy:
                    strategy_translated = translate_strategy_name(strategy)
                    if config_pretty.startswith(strategy_translated):
                        # Extract just the params part
                        config_parts = config_pretty.split(" | ", 1)
                        if len(config_parts) > 1:
                            run_short_names[run_display] = f"{strategy_short}: {config_parts[1]}"
                        else:
                            run_short_names[run_display] = strategy_short
                    else:
                        run_short_names[run_display] = config_pretty
                else:
                    run_short_names[run_display] = config_pretty
            else:
                # Standard format: just use strategy
                run_short_names[run_display] = strategy_short

    table_data = []

    for problem_id in sorted(reference_problems):
        row = {"Problem": problem_id}

        for run_display in summaries.keys():
            summary = summaries[run_display]
            short_name = run_short_names[run_display]

            if problem_id not in summary:
                # Problem not in this run
                row[f"{short_name}_Seeds"] = "-"
                row[f"{short_name}_Calls"] = "-"
                row[f"{short_name}_SFLOPs"] = "-"
                row[f"{short_name}_Lemmas"] = "-"
            else:
                problem_data = summary[problem_id]
                seeds_solved = problem_data.get("seeds_solved", 0)
                total_seeds = problem_data.get("total_seeds", 0)
                avg_calls = problem_data.get("avg_prover_calls", problem_data.get("avg_attempts", 0))
                avg_sflops = problem_data.get("avg_sflops", 0)
                avg_used_lemmas = problem_data.get("avg_used_lemmas", None)

                # Format seeds as "X/Y"
                if seeds_solved == 0:
                    row[f"{short_name}_Seeds"] = f"0/{total_seeds}"
                else:
                    row[f"{short_name}_Seeds"] = f"{seeds_solved}/{total_seeds}"

                # Format calls
                if avg_calls == -1 or seeds_solved == 0:
                    row[f"{short_name}_Calls"] = "-"
                else:
                    row[f"{short_name}_Calls"] = f"{avg_calls:.2f}"

                # Format SFLOPs
                if avg_sflops == -1 or seeds_solved == 0:
                    row[f"{short_name}_SFLOPs"] = "-"
                else:
                    row[f"{short_name}_SFLOPs"] = f"{avg_sflops:.0f}"

                # Format used lemmas
                if avg_used_lemmas is None or seeds_solved == 0:
                    row[f"{short_name}_Lemmas"] = "-"
                else:
                    row[f"{short_name}_Lemmas"] = f"{avg_used_lemmas:.2f}"

        table_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(table_data)

    # Display with horizontal scroll
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=600
    )

    # Show legend for short names
    with st.expander("Run Name Legend"):
        legend_data = [{"Short Name": short, "Full Path": full} for full, short in run_short_names.items()]
        legend_df = pd.DataFrame(legend_data)
        st.dataframe(legend_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Summary statistics
    st.subheader("Summary Statistics")

    summary_stats = []
    for run_display, summary in summaries.items():
        total_problems = len([p for p in reference_problems if p in summary])
        problems_solved = len([p for p in reference_problems if p in summary and summary[p].get("seeds_solved", 0) > 0])

        total_calls = sum(
            summary[p].get("avg_attempts", 0)
            for p in reference_problems
            if p in summary and summary[p].get("avg_attempts", -1) > 0
        )
        avg_calls = total_calls / problems_solved if problems_solved > 0 else 0

        total_sflops = sum(
            summary[p].get("avg_sflops", 0)
            for p in reference_problems
            if p in summary and summary[p].get("avg_sflops", -1) > 0
        )
        avg_sflops = total_sflops / problems_solved if problems_solved > 0 else 0

        summary_stats.append({
            "Run": run_display,
            "Problems Solved": f"{problems_solved}/{total_problems}",
            "Solve Rate": f"{(problems_solved/total_problems*100):.1f}%" if total_problems > 0 else "0%",
            "Avg Prover Calls": f"{avg_calls:.2f}",
            "Avg SFLOPs": f"{avg_sflops:.0f}"
        })

    summary_df = pd.DataFrame(summary_stats)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Export option
    st.markdown("---")
    st.subheader("Export")

    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Comparison Table as CSV",
        data=csv_data,
        file_name="simulation_comparison.csv",
        mime="text/csv"
    )
