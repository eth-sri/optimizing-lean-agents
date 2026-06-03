"""
Utility module for browsing and selecting runs from the scratch folder.

Provides functionality to discover available runs and format them in a user-friendly way.
"""
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime


def get_available_runs(base_path: str = None, include_minified: bool = True) -> List[Tuple[str, str, str]]:
    """
    Scan the scratch folder and return available run directories.

    Args:
        base_path: Path to the base scratch results directory. If None, looks relative to the app location.
        include_minified: If True, also search scratch/dump/ for minified data

    Returns:
        List of tuples (display_name, full_path, data_type) sorted by date descending
        where data_type is either 'full' or 'minified'
    """
    if base_path is None:
        # Try to find relative to the app directory
        base_path = Path(__file__).parent.parent.parent / "scratch" / "results" / "combined"

    runs = []

    # Search scratch/results/combined for both full results and dump/ subdirectories
    combined_base = Path(__file__).parent.parent.parent / "scratch" / "results" / "combined"
    if combined_base.exists():
        runs.extend(_scan_combined_results_directory(combined_base))

    # Also search legacy scratch/dump directory for backwards compatibility
    dump_base = Path(__file__).parent.parent.parent / "scratch" / "dump"
    if dump_base.exists():
        runs.extend(_scan_results_directory(dump_base, "minified"))

        # Check for special combined_dump directory
        combined_dump = dump_base / "combined_dump"
        if combined_dump.exists():
            display_name = "Combined Dump (All Runs)"
            # Point to the minified/round0 directory
            runs.append((display_name, str(combined_dump / "minified"), "minified"))

    # Search scratch/dump_updated directory for updated dumps
    dump_updated_base = Path(__file__).parent.parent.parent / "scratch" / "dump_updated"
    if dump_updated_base.exists():
        runs.extend(_scan_results_directory(dump_updated_base, "minified"))

        # Check for special combined_dump directory in dump_updated
        updated_combined_dump = dump_updated_base / "combined_dump"
        if updated_combined_dump.exists():
            display_name = "⭐ Updated Combined Dump (Recommended)"
            runs.insert(0, (display_name, str(updated_combined_dump), "minified"))

    # Search outputs directory for named experiment runs
    outputs_base = Path(__file__).parent.parent.parent / "outputs"
    if outputs_base.exists():
        runs.extend(_scan_outputs_directory(outputs_base))

    # Sort by timestamp (descending)
    runs.sort(key=lambda x: x[0], reverse=True)

    return runs


def _scan_combined_results_directory(base: Path) -> List[Tuple[str, str, str]]:
    """
    Scan scratch/results/combined directory for both full results and dump/ subdirectories.

    Handles two structures:
    1. New structure: YYYY/MM/DD/HHMMSS/dump/minified (and dump/hierarchical)
    2. Legacy structure: YYYY/MM/DD/HHMMSS/round0 (full results)

    Args:
        base: Base directory to scan (scratch/results/combined)

    Returns:
        List of tuples (display_name, full_path, data_type)
    """
    runs = []

    # Find all directories in the pattern: YYYY/MM/DD/HHMMSS
    for year_dir in sorted(base.glob("*"), reverse=True):
        if not year_dir.is_dir():
            continue

        for month_dir in sorted(year_dir.glob("*"), reverse=True):
            if not month_dir.is_dir():
                continue

            for day_dir in sorted(month_dir.glob("*"), reverse=True):
                if not day_dir.is_dir():
                    continue

                for time_dir in sorted(day_dir.glob("*"), reverse=True):
                    if not time_dir.is_dir():
                        continue

                    # Format: 2025/10/28 10:13:19
                    year = year_dir.name
                    month = month_dir.name
                    day = day_dir.name
                    time_str = time_dir.name

                    # Parse time string (HHMMSS format)
                    if len(time_str) == 6 and time_str.isdigit():
                        hour = time_str[0:2]
                        minute = time_str[2:4]
                        second = time_str[4:6]
                        display_name = f"{year}/{month}/{day} {hour}:{minute}:{second}"
                    else:
                        # Skip if not valid timestamp format
                        continue

                    # Check for new structure: dump/minified (hierarchical is automatically found)
                    dump_dir = time_dir / "dump"
                    minified_dir = dump_dir / "minified"

                    if minified_dir.exists():
                        # New structure with dump/minified subdirectory
                        # (hierarchical sibling will be auto-discovered by app.py)
                        runs.append((f"{display_name} (minified)", str(minified_dir), "minified"))
                    elif (time_dir / "round0").exists():
                        # Legacy structure: full results with round0/ directory
                        runs.append((f"{display_name} (full)", str(time_dir), "full"))

    return runs


def _scan_results_directory(base: Path, data_type: str) -> List[Tuple[str, str, str]]:
    """
    Internal helper to scan a results directory for YYYY/MM/DD/HHMMSS pattern.

    Args:
        base: Base directory to scan
        data_type: Either 'full' or 'minified'

    Returns:
        List of tuples (display_name, full_path, data_type)
    """
    runs = []

    # Find all directories in the pattern: YYYY/MM/DD/HHMMSS
    for year_dir in sorted(base.glob("*"), reverse=True):
        if not year_dir.is_dir():
            continue

        for month_dir in sorted(year_dir.glob("*"), reverse=True):
            if not month_dir.is_dir():
                continue

            for day_dir in sorted(month_dir.glob("*"), reverse=True):
                if not day_dir.is_dir():
                    continue

                for time_dir in sorted(day_dir.glob("*"), reverse=True):
                    if not time_dir.is_dir():
                        continue

                    # For full results: just the timestamp dir
                    # For minified: need to check for minified/ subdirectory
                    if data_type == "minified":
                        minified_dir = time_dir / "minified"
                        if not minified_dir.exists():
                            continue
                        target_dir = minified_dir
                    else:
                        target_dir = time_dir

                    # Format: 2025/10/28 10:13:19
                    year = year_dir.name
                    month = month_dir.name
                    day = day_dir.name
                    time_str = time_dir.name

                    # Parse time string (HHMMSS format)
                    if len(time_str) == 6 and time_str.isdigit():
                        hour = time_str[0:2]
                        minute = time_str[2:4]
                        second = time_str[4:6]

                        # Create display name
                        display_name = f"{year}/{month}/{day} {hour}:{minute}:{second}"
                        full_path = str(target_dir)

                        runs.append((display_name, full_path, data_type))

    return runs


def _scan_outputs_directory(base: Path) -> List[Tuple[str, str, str]]:
    """
    Scan outputs directory for named experiment runs.

    Handles structure: outputs/{dataset}/{prover_type}/runs/{run_name}/YYYY/MM/DD/HHMMSS/combined/results/dump/minified

    Args:
        base: Base outputs directory

    Returns:
        List of tuples (display_name, full_path, data_type)
    """
    runs = []

    # Scan outputs/{dataset}/{prover_type}/runs/{run_name}/
    for dataset_dir in base.iterdir():
        if not dataset_dir.is_dir():
            continue

        for prover_type_dir in dataset_dir.iterdir():
            if not prover_type_dir.is_dir():
                continue

            runs_dir = prover_type_dir / "runs"
            if not runs_dir.exists():
                continue

            for run_name_dir in runs_dir.iterdir():
                if not run_name_dir.is_dir():
                    continue

                run_name = run_name_dir.name
                dataset = dataset_dir.name

                # Scan for YYYY/MM/DD/HHMMSS/combined/results/dump/minified
                for year_dir in sorted(run_name_dir.glob("*"), reverse=True):
                    if not year_dir.is_dir() or not year_dir.name.isdigit():
                        continue

                    for month_dir in sorted(year_dir.glob("*"), reverse=True):
                        if not month_dir.is_dir():
                            continue

                        for day_dir in sorted(month_dir.glob("*"), reverse=True):
                            if not day_dir.is_dir():
                                continue

                            for time_dir in sorted(day_dir.glob("*"), reverse=True):
                                if not time_dir.is_dir():
                                    continue

                                # Check for combined/results/dump/minified structure
                                minified_dir = time_dir / "combined" / "results" / "dump" / "minified"
                                if not minified_dir.exists():
                                    continue

                                # Format timestamp
                                time_str = time_dir.name
                                if len(time_str) == 6 and time_str.isdigit():
                                    hour = time_str[0:2]
                                    minute = time_str[2:4]
                                    second = time_str[4:6]
                                    timestamp = f"{year_dir.name}/{month_dir.name}/{day_dir.name} {hour}:{minute}:{second}"
                                else:
                                    continue

                                # Create descriptive display name
                                display_name = f"📦 {dataset}/{run_name} ({timestamp})"
                                runs.append((display_name, str(minified_dir), "minified"))

    return runs


def format_runs_for_dropdown(runs: List[Tuple[str, str, str]]) -> Tuple[List[str], dict, dict]:
    """
    Format runs for use in a Streamlit selectbox dropdown.

    Args:
        runs: List of (display_name, full_path, data_type) tuples

    Returns:
        Tuple of (display_names_list, path_mapping_dict, data_type_mapping_dict)
    """
    if not runs:
        return [], {}, {}

    display_names = [name for name, _, _ in runs]
    path_mapping = {name: path for name, path, _ in runs}
    data_type_mapping = {name: dtype for name, _, dtype in runs}

    return display_names, path_mapping, data_type_mapping


def get_run_info(run_path: str) -> dict:
    """
    Get metadata about a run directory.

    Args:
        run_path: Path to the run directory

    Returns:
        Dictionary with run info (date, time, relative_path)
    """
    path = Path(run_path)

    try:
        # Get file modification time as run creation time
        mtime = path.stat().st_mtime
        run_time = datetime.fromtimestamp(mtime)

        # Get relative path from scratch base
        try:
            scratch_base = Path(__file__).parent.parent.parent / "scratch" / "results" / "combined"
            relative = path.relative_to(scratch_base)
        except (ValueError, AttributeError):
            relative = path.name

        return {
            "path": run_path,
            "relative_path": str(relative),
            "created": run_time.isoformat(),
            "display_time": run_time.strftime("%Y/%m/%d %H:%M:%S")
        }
    except Exception as e:
        return {
            "path": run_path,
            "error": str(e)
        }
