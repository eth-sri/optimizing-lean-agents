"""Unified factory for building SimulatedProblems from data sources."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .full_proof import load_full_proof_data
from .agent import load_agent_data
from .types import AttemptPair, BreakdownTemplate
from ..actions import DetailedCost


def load_problems(
    full_proof_sources: Optional[Dict[str, str]] = None,
    agent_config: Optional[Dict] = None,
    seed: int = 42,
    load_code: bool = False,
) -> List['SimulatedProblem']:
    """Build SimulatedProblem list from data sources.

    Args:
        full_proof_sources: {model_name: minified_json_path}
            e.g., {"8b": "outputs/putnam/full_proof_8b/minified_8b.json",
                    "32b": "outputs/putnam/full_proof_32b/minified_32b.json"}
        agent_config: Agent config dict with keys:
            - "sources": {model_name: run_dir_path}
            - "shared_breakdowns": bool (default False)
        seed: Random seed for initial shuffling

    Returns:
        List of SimulatedProblem instances
    """
    from ..target import TargetNode
    from ..problem import SimulatedProblem

    # 1. Load full proof data per model
    full_proof_by_problem: Dict[str, Dict[str, List[AttemptPair]]] = {}
    if full_proof_sources:
        for model_name, path in full_proof_sources.items():
            model_data = load_full_proof_data(path, model_name)
            for pid, pairs in model_data.items():
                if pid not in full_proof_by_problem:
                    full_proof_by_problem[pid] = {}
                full_proof_by_problem[pid][model_name] = pairs

    # 2. Load agent data from all sources
    agent_data_by_problem: Dict[str, Dict] = {}
    if agent_config:
        sources = agent_config.get("sources", {})
        shared_breakdowns = agent_config.get("shared_breakdowns", False)
        agent_data_by_problem = _load_multi_agent_sources(sources, shared_breakdowns, load_code=load_code)

    # 3. Collect all problem IDs
    all_pids = set(full_proof_by_problem.keys()) | set(agent_data_by_problem.keys())

    # 4. Build TargetNode tree for each problem
    problems = []
    for pid in sorted(all_pids):
        # Full proof data for root-level PROVE actions
        proof_data = full_proof_by_problem.get(pid, {})

        # Agent breakdown templates for DECOMPOSE/CREATE_BREAKDOWN
        agent_info = agent_data_by_problem.get(pid, {})
        breakdown_templates = agent_info.get('breakdown_templates', [])

        # Build root TargetNode
        root = TargetNode(
            target_id=pid,
            target_type="problem",
            proof_data=proof_data,
            breakdown_templates=breakdown_templates,
            seed=seed,
        )

        problems.append(SimulatedProblem(
            problem_id=pid,
            root=root,
        ))

    return problems


def _load_multi_agent_sources(
    sources: Dict[str, str],
    shared_breakdowns: bool,
    load_code: bool = False,
) -> Dict[str, Dict]:
    """Load agent data from multiple sources, optionally merging shared breakdowns.

    Args:
        sources: {model_name: run_dir_path}
        shared_breakdowns: If True, merge breakdown templates that share the same
            (round_id, breakdown_id) key across sources.

    Returns:
        Dict mapping problem_id to {'breakdown_templates': List[BreakdownTemplate]}
    """
    if not sources:
        return {}

    # Load each source independently
    per_source: List[Dict[str, Dict]] = []
    for model_name, path in sources.items():
        source_data = load_agent_data(path, load_code=load_code)
        per_source.append(source_data)

    # Collect all problem IDs across sources
    all_pids = set()
    for source_data in per_source:
        all_pids.update(source_data.keys())

    result: Dict[str, Dict] = {}
    for pid in sorted(all_pids):
        # Gather templates from each source for this problem
        all_templates: List[List[BreakdownTemplate]] = []
        for source_data in per_source:
            if pid in source_data:
                templates = source_data[pid].get('breakdown_templates', [])
                all_templates.append(templates)

        if not all_templates:
            continue

        if shared_breakdowns and len(all_templates) > 1:
            merged = _merge_breakdown_templates(all_templates)
        else:
            # Concatenate all templates
            merged = []
            for templates in all_templates:
                merged.extend(templates)

        result[pid] = {'breakdown_templates': merged}

    return result


def _merge_breakdown_templates(
    per_source_templates: List[List[BreakdownTemplate]],
) -> List[BreakdownTemplate]:
    """Merge breakdown templates from multiple sources by breakdown_key.

    Templates with the same (round_id, breakdown_id) key are merged:
    their target_proof_data dicts are combined (keys are model names
    like "8b"/"32b" so they won't collide), and costs are summed.

    Templates without a breakdown_key or with unique keys are kept as-is.
    """
    # Group by breakdown_key
    by_key: Dict[Tuple[int, int], List[BreakdownTemplate]] = defaultdict(list)
    no_key: List[BreakdownTemplate] = []

    for templates in per_source_templates:
        for tmpl in templates:
            if tmpl.breakdown_key is not None:
                by_key[tmpl.breakdown_key].append(tmpl)
            else:
                no_key.append(tmpl)

    merged: List[BreakdownTemplate] = []

    # Merge templates that share a breakdown_key
    for key in sorted(by_key.keys()):
        group = by_key[key]
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Merge: combine target_proof_data; use first source's cost (one-time cost, don't sum)
        combined_proof_data: Dict[int, Dict[str, List[AttemptPair]]] = {}
        combined_cost = group[0].cost

        for tmpl in group:
            for target_id, model_data in tmpl.target_proof_data.items():
                if target_id not in combined_proof_data:
                    combined_proof_data[target_id] = {}
                for model_name, pairs in model_data.items():
                    if model_name not in combined_proof_data[target_id]:
                        combined_proof_data[target_id][model_name] = []
                    combined_proof_data[target_id][model_name].extend(pairs)

        merged.append(BreakdownTemplate(
            breakdown_idx=group[0].breakdown_idx,
            cost=combined_cost,
            target_proof_data=combined_proof_data,
            breakdown_key=key,
        ))

    # Add templates without a key
    merged.extend(no_key)

    return merged
