"""
Utilities for consolidating full_records across rounds and iterations.

This module provides shared consolidation logic used by both:
- RecursiveProverComponent (after each round)

Also handles SFLOP calculation by reading effective_config.yaml and enriching
records with model_config_path and calculated SFLOPs for each component.
"""

import os
import glob
import yaml
from collections import defaultdict
from typing import List, Dict, Any, Optional
from loguru import logger
from jload import jload, jsave
from metadata_utils import get_origin_problem_id
from seed_data_models.model_config import (
    get_effective_parameters,
    calculate_sflops_from_tokens,
)


def consolidate_to_base_full_records(
    base_output_dir: str,
    round_num: int = None,
    verbosity: int = 3,
    add_sflops: bool = True
) -> str:
    """
    Consolidate full_records from round(s) to base-level full_records directory.

    This function loads records from round{N}/prover/full_records/ directories and
    consolidates them into {base_output_dir}/full_records/ grouped by problem_id.

    Enriches records with:
    - Compilation data (pass, complete, errors, warnings) from code_compilation_repl.json
    - Model config paths and calculated SFLOPs (if add_sflops=True)

    If round_num is specified, only consolidates up to and including that round.
    If round_num is None, consolidates all available rounds.

    This enables the feedback loop: after each round completes, the base-level
    full_records/ is updated so the next round's FeedbackDataLoaderComponent
    can access the results.

    Args:
        base_output_dir: Base output directory containing round directories
        round_num: If specified, only consolidate up to this round (inclusive).
                   If None, consolidate all rounds.
        verbosity: Logging verbosity level
        add_sflops: If True, loads effective_config.yaml and adds SFLOPs to records

    Returns:
        Path to the consolidated full_records directory
    """
    if verbosity >= 2:
        if round_num is not None:
            logger.info(f"Consolidating full_records up to round {round_num}...")
        else:
            logger.info("Consolidating full_records from all rounds...")

    # Load effective config and extract model paths if SFLOP enrichment is enabled
    model_config_paths = {}
    if add_sflops:
        effective_config = load_effective_config(base_output_dir, verbosity=verbosity)
        if effective_config:
            model_config_paths = extract_model_config_paths(effective_config, verbosity=verbosity)

    # Discover round directories
    if round_num is not None:
        # Only include rounds up to round_num
        round_dirs = []
        for r in range(round_num + 1):
            round_dir = os.path.join(base_output_dir, f"round{r}")
            if os.path.isdir(round_dir):
                round_dirs.append(round_dir)
    else:
        # Include all round directories
        round_dirs = sorted([
            d for d in glob.glob(os.path.join(base_output_dir, "round*"))
            if os.path.isdir(d)
        ])

    if not round_dirs:
        if verbosity >= 1:
            logger.warning(f"No round directories found in: {base_output_dir}")
        return None

    if verbosity >= 2:
        round_nums = [os.path.basename(d) for d in round_dirs]
        logger.info(f"  Loading from rounds: {', '.join(round_nums)}")

    # Load all records from all specified rounds, grouped by problem
    records_by_problem = defaultdict(list)
    total_records = 0

    for round_dir in round_dirs:
        round_name = os.path.basename(round_dir)
        prover_dir = os.path.join(round_dir, "prover")
        full_records_dir = os.path.join(prover_dir, "full_records")

        if not os.path.exists(full_records_dir):
            if verbosity >= 2:
                logger.warning(f"  {round_name}: full_records directory not found, skipping")
            continue

        # Load all JSON files from this round's full_records directory
        json_files = sorted([
            f for f in os.listdir(full_records_dir)
            if f.endswith('.json')
        ])

        if not json_files:
            if verbosity >= 2:
                logger.warning(f"  {round_name}: No JSON files found, skipping")
            continue

        # Build compilation map from iterations
        compilation_map = _build_compilation_map(prover_dir, verbosity)

        round_records = 0
        for json_file in json_files:
            file_path = os.path.join(full_records_dir, json_file)
            records = jload(file_path)

            if not isinstance(records, list):
                if verbosity >= 1:
                    logger.warning(f"  {round_name}/{json_file}: Unexpected format, expected list")
                continue

            # Enrich records with compilation data
            enriched_records = _enrich_records_with_compilation(records, compilation_map)

            # Enrich records with SFLOPs if model config paths are available
            if model_config_paths:
                enriched_records = enrich_records_with_sflops(
                    enriched_records,
                    model_config_paths,
                    base_output_dir,
                    verbosity=verbosity
                )

            # Extract problem_id from filename (without .json extension)
            problem_id = json_file[:-5]

            # Add to grouped collection
            records_by_problem[problem_id].extend(enriched_records)
            round_records += len(enriched_records)

        total_records += round_records
        if verbosity >= 2:
            logger.info(f"  {round_name}: loaded {round_records} records from {len(json_files)} problems")

    if total_records == 0:
        if verbosity >= 1:
            logger.warning("No records found to consolidate")
        return None

    # Save consolidated records to base-level full_records/
    consolidated_records_dir = os.path.join(base_output_dir, "full_records")
    os.makedirs(consolidated_records_dir, exist_ok=True)

    problems_saved = 0
    for problem_id, records in records_by_problem.items():
        output_path = os.path.join(consolidated_records_dir, f"{problem_id}.json")
        jsave(records, output_path)
        problems_saved += 1

    if verbosity >= 1:
        logger.info(f"  Consolidated {total_records} records from {problems_saved} problems")
        logger.info(f"  Saved to: {consolidated_records_dir}")

    return consolidated_records_dir


def _build_compilation_map(prover_dir: str, verbosity: int = 3) -> Dict[str, Dict[str, Any]]:
    """
    Build a map of uid -> compilation result from all iterations in prover_dir.

    Scans iter*/code_compilation_repl*.json files and creates a dict keyed by uid.
    """
    compilation_map = {}

    # Find all iterations
    iter_dirs = sorted([
        d for d in glob.glob(os.path.join(prover_dir, "iter*"))
        if os.path.isdir(d)
    ])

    for iter_dir in iter_dirs:
        # Load initial compilation results
        compilation_file = os.path.join(iter_dir, "code_compilation_repl.json")
        if os.path.exists(compilation_file):
            try:
                compilation_results = jload(compilation_file)
                if isinstance(compilation_results, list):
                    for comp in compilation_results:
                        if isinstance(comp, dict) and "name" in comp:
                            compilation_map[comp["name"]] = comp
            except Exception as e:
                if verbosity >= 2:
                    logger.warning(f"Error loading {compilation_file}: {e}")

        # Load correction round compilation results
        corr_round = 1
        while True:
            compilation_corr_file = os.path.join(iter_dir, f"code_compilation_repl_corr{corr_round}.json")
            if os.path.exists(compilation_corr_file):
                try:
                    compilation_results = jload(compilation_corr_file)
                    if isinstance(compilation_results, list):
                        for comp in compilation_results:
                            if isinstance(comp, dict) and "name" in comp:
                                compilation_map[comp["name"]] = comp
                except Exception as e:
                    if verbosity >= 2:
                        logger.warning(f"Error loading {compilation_corr_file}: {e}")
                corr_round += 1
            else:
                break

    return compilation_map


def _enrich_records_with_compilation(records: List[Dict[str, Any]], compilation_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich records with compilation data from compilation_map.

    For each record with a matching uid in the compilation_map, adds:
    - compilation_result (pass, complete, errors, warnings, etc.)
    - compilation_summary (error classification and error_counts)
    - verify_time

    Args:
        records: List of records to enrich
        compilation_map: Map of uid -> compilation data

    Returns:
        List of enriched records
    """
    enriched = []
    for record in records:
        uid = record.get("uid")
        if uid and uid in compilation_map:
            comp = compilation_map[uid]
            # Add compilation data if not already present
            if "compilation_result" not in record:
                record["compilation_result"] = comp.get("compilation_result", {})
            if "compilation_summary" not in record:
                record["compilation_summary"] = comp.get("compilation_summary")
            if "verify_time" not in record:
                record["verify_time"] = comp.get("verify_time")
        enriched.append(record)
    return enriched


def load_base_full_records(base_output_dir: str, verbosity: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load existing base-level full_records grouped by problem_id.

    Args:
        base_output_dir: Base output directory
        verbosity: Logging verbosity level

    Returns:
        Dictionary mapping problem_id to list of records
    """
    full_records_dir = os.path.join(base_output_dir, "full_records")

    if not os.path.exists(full_records_dir):
        if verbosity >= 2:
            logger.info("No existing base-level full_records directory")
        return {}

    records_by_problem = {}
    json_files = [f for f in os.listdir(full_records_dir) if f.endswith('.json')]

    for json_file in json_files:
        file_path = os.path.join(full_records_dir, json_file)
        records = jload(file_path)

        if isinstance(records, list):
            problem_id = json_file[:-5]  # Remove .json
            records_by_problem[problem_id] = records

    if verbosity >= 2 and records_by_problem:
        total_records = sum(len(records) for records in records_by_problem.values())
        logger.info(f"Loaded {total_records} existing records from {len(records_by_problem)} problems")

    return records_by_problem


def load_effective_config(base_output_dir: str, verbosity: int = 3) -> Optional[Dict[str, Any]]:
    """
    Load effective_config.yaml from the configs directory.

    Args:
        base_output_dir: Base output directory containing configs/
        verbosity: Logging verbosity level

    Returns:
        Parsed YAML config dict, or None if not found
    """
    config_path = os.path.join(base_output_dir, "configs", "effective_config.yaml")

    if not os.path.exists(config_path):
        if verbosity >= 2:
            logger.warning(f"effective_config.yaml not found at {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if verbosity >= 3:
            logger.info(f"Loaded effective_config from {config_path}")
        return config
    except Exception as e:
        if verbosity >= 1:
            logger.warning(f"Error loading effective_config.yaml: {e}")
        return None


def extract_model_config_paths(config: Dict[str, Any], verbosity: int = 3) -> Dict[str, str]:
    """
    Extract model_config_paths for each component from effective_config.

    Maps component names to their model_config_path. For inline configs,
    resolves the model name to a path if possible.

    Returns:
        Dict mapping component -> model_config_path
        Examples: {"breakdown": "configs/models/openai/oss-20b-high.yaml", ...}
    """
    paths = {}

    if not config or "components" not in config:
        return paths

    components = config["components"]

    # Breakdown component
    if "breakdown" in components and "config" in components["breakdown"]:
        breakdown_cfg = components["breakdown"]["config"]
        if "model_config" in breakdown_cfg:
            model_cfg = breakdown_cfg["model_config"]
            if isinstance(model_cfg, str):
                # It's a path
                paths["breakdown"] = model_cfg
            elif isinstance(model_cfg, dict) and "model" in model_cfg:
                # It's an inline config, try to resolve model name to path
                model_name = model_cfg["model"]
                # Try common patterns
                if "goedel" in model_name.lower() and "formalizer" in model_name.lower():
                    if "32b" in model_name.lower() or "32" in model_name:
                        paths["breakdown"] = "configs/models/goedel_formalizer_v2/32b.yaml"
                    elif "8b" in model_name.lower() or "8" in model_name:
                        paths["breakdown"] = "configs/models/goedel_formalizer_v2/8b.yaml"
                elif "gpt-oss" in model_name.lower() or "openai" in model_name.lower():
                    if "120b" in model_name.lower():
                        paths["breakdown"] = "configs/models/openai/oss-120b-high.yaml"
                    elif "20b" in model_name.lower():
                        paths["breakdown"] = "configs/models/openai/oss-20b-high.yaml"

    # Breakdown parser component
    if "breakdown_parser" in components and "config" in components["breakdown_parser"]:
        parser_cfg = components["breakdown_parser"]["config"]
        if "model_config" in parser_cfg:
            model_cfg = parser_cfg["model_config"]
            if isinstance(model_cfg, str):
                paths["breakdown_parser"] = model_cfg

    # Formalization component
    if "formalizer" in components and "config" in components["formalizer"]:
        form_cfg = components["formalizer"]["config"]
        if "model_config" in form_cfg:
            model_cfg = form_cfg["model_config"]
            if isinstance(model_cfg, str):
                paths["formalization"] = model_cfg
            elif isinstance(model_cfg, dict) and "model" in model_cfg:
                model_name = model_cfg["model"]
                if "formalizer" in model_name.lower():
                    if "32b" in model_name.lower() or "32" in model_name:
                        paths["formalization"] = "configs/models/goedel_formalizer_v2/32b.yaml"
                    elif "8b" in model_name.lower() or "8" in model_name:
                        paths["formalization"] = "configs/models/goedel_formalizer_v2/8b.yaml"

    if verbosity >= 3:
        logger.info(f"Extracted model_config_paths: {paths}")

    return paths


def enrich_records_with_sflops(
    records: List[Dict[str, Any]],
    model_config_paths: Dict[str, str],
    base_output_dir: str,
    verbosity: int = 3
) -> List[Dict[str, Any]]:
    """
    Enrich records with model_config_path and calculated SFLOPs for each component.

    For each component that has output_tokens in detailed_cost, calculates SFLOPs
    and adds them to the detailed_cost dict. Also adds model_config_path metadata.

    Args:
        records: List of records to enrich
        model_config_paths: Dict mapping component -> model_config_path
        base_output_dir: Base output directory (for resolving relative paths)
        verbosity: Logging verbosity level

    Returns:
        List of enriched records
    """
    enriched = []
    missing_params = set()

    for record in records:
        # Determine component from metadata or context
        lemma_id = record.get("metadata", {}).get("lemma_id", -1) if isinstance(record.get("metadata"), dict) else -1

        # Try to identify component from record structure
        component = None
        if "breakdown_prompt" in record:
            component = "breakdown"
        elif record.get("name", "").endswith("_breakdown_parser") or "parsed_breakdown" in record:
            component = "breakdown_parser"
        elif "formalization_reasoning" in record or "formal_statement" in record:
            component = "formalization"

        # Add model_config_path if we know the component
        if component and component in model_config_paths:
            record["model_config_path"] = model_config_paths[component]

            # Calculate SFLOPs for this component
            detailed_cost = record.get("detailed_cost", {})
            if detailed_cost and "output_tokens" in detailed_cost:
                output_tokens = int(detailed_cost["output_tokens"])
                model_path = model_config_paths[component]

                # Make path absolute
                if not os.path.isabs(model_path):
                    model_path = os.path.join(base_output_dir, model_path)

                sflops = calculate_sflops_from_tokens(output_tokens, model_path, base_path=base_output_dir)

                # Add sflops to detailed_cost
                if "sflops" not in detailed_cost:
                    detailed_cost["sflops"] = sflops
                    record["detailed_cost"] = detailed_cost

                    # Check if we're missing effective_parameters
                    if sflops == output_tokens:  # Fallback happened
                        eff_params = get_effective_parameters(model_path, base_path=base_output_dir)
                        if eff_params is None:
                            missing_params.add(model_path)

        enriched.append(record)

    if missing_params and verbosity >= 1:
        logger.warning(f"Missing effective_parameters for models: {missing_params}")

    return enriched
