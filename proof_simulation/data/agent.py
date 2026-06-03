"""Load agent (seed prover) Session data and convert to simulation types."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .types import AttemptData, AttemptPair, BreakdownTemplate
from ..actions import DetailedCost


def extract_model_name(model_config_path: Optional[str]) -> Optional[str]:
    """Extract model name from model_config_path.

    Examples:
        'configs/hydra/prover/models/goedel_prover_v2/sft8b.yaml' -> '8b'
        'configs/hydra/prover/models/goedel_prover_v2/32b.yaml' -> '32b'
    """
    if not model_config_path:
        return None
    match = re.search(r'(\d+b)', model_config_path.lower())
    return match.group(1) if match else None


def load_agent_data_flat(run_dir: str, model_name: str, load_code: bool = False) -> Dict[str, List[AttemptPair]]:
    """Load agent Session data and flatten all proof attempts into AttemptPairs.

    Returns the same format as load_full_proof_data: {problem_id: [AttemptPair]}.
    Filters to only attempts matching the given model_name.
    """
    from seed_data_models.session import Session

    run_path = Path(run_dir)
    session = Session.load_from_minified(run_path, verbose=False, load_proof_code=load_code)

    result: Dict[str, List[AttemptPair]] = {}
    for pid, problem in session.problems.items():
        pairs: List[AttemptPair] = []
        for breakdown in problem.breakdowns.values():
            if not breakdown.parsed_breakdown:
                continue
            pb = breakdown.parsed_breakdown
            # Collect from theorem
            _collect_pairs(pb.theorem.formalizations, model_name, pairs, load_code=load_code)
            # Collect from lemmas
            for lemma in pb.lemmas.values():
                _collect_pairs(lemma.formalizations, model_name, pairs, load_code=load_code)
        if pairs:
            result[pid] = pairs

    return result


def _collect_pairs(formalizations, model_name: str, pairs: List[AttemptPair], load_code: bool = False):
    """Collect AttemptPairs for a specific model from formalizations into pairs list."""
    all_attempts = []
    for form in formalizations:
        for attempt in form.proof_attempts:
            mn = extract_model_name(attempt.model_config_path)
            if mn == model_name:
                all_attempts.append(attempt)
    if all_attempts:
        pairs.extend(_pair_initials_with_corrections(all_attempts, load_code=load_code))


def load_agent_data(run_dir: str, load_code: bool = False) -> Dict[str, Dict[str, object]]:
    """Load agent Session data from minified directory.

    Returns:
        Dict mapping problem_id to dict with:
            - 'breakdown_templates': List[BreakdownTemplate]
            - 'full_proof_data': Dict[str, List[AttemptPair]]  (empty, agent has no full proofs)
    """
    from seed_data_models.session import Session

    run_path = Path(run_dir)
    session = Session.load_from_minified(run_path, verbose=False, load_proof_code=load_code)

    result = {}
    for pid, problem in session.problems.items():
        templates = _build_breakdown_templates(problem, load_code=load_code)
        result[pid] = {
            'breakdown_templates': templates,
        }

    return result


def _build_breakdown_templates(problem, load_code: bool = False) -> List[BreakdownTemplate]:
    """Build BreakdownTemplates from a Problem's breakdowns."""
    templates = []

    # Sort breakdowns by breakdown_id for deterministic ordering
    sorted_breakdowns = sorted(
        problem.breakdowns.values(),
        key=lambda b: (b.round_id, b.breakdown_id)
    )

    for idx, breakdown in enumerate(sorted_breakdowns):
        if not breakdown.parsed_breakdown:
            continue

        # Calculate breakdown creation cost (everything except prover attempts)
        cost = _compute_breakdown_cost(breakdown.get_total_cost)

        # Build per-target proof data
        target_proof_data: Dict[int, Dict[str, List[AttemptPair]]] = {}
        pb = breakdown.parsed_breakdown

        # Theorem (-1)
        theorem_data = _extract_proof_data(pb.theorem.formalizations, load_code=load_code)
        if theorem_data:
            target_proof_data[-1] = theorem_data

        # Lemmas
        for lemma_id, lemma in pb.lemmas.items():
            lemma_data = _extract_proof_data(lemma.formalizations, load_code=load_code)
            if lemma_data:
                target_proof_data[lemma_id] = lemma_data

        templates.append(BreakdownTemplate(
            breakdown_idx=idx,
            cost=cost,
            target_proof_data=target_proof_data,
            breakdown_key=(breakdown.round_id, breakdown.breakdown_id),
        ))

    return templates


def _compute_breakdown_cost(get_total_cost_fn) -> DetailedCost:
    """Compute the cost of creating a breakdown using Breakdown.get_total_cost.

    Uses get_total_cost(cost_type, exclude_prover_calls=True) which correctly
    sums breakdown + parser + formalization costs with proper per-model
    effective_params for sflops calculation.
    """
    return DetailedCost(
        input_sflops=int(get_total_cost_fn('input_sflops', exclude_prover_calls=True)),
        output_sflops=int(get_total_cost_fn('output_sflops', exclude_prover_calls=True)),
        input_tokens=int(get_total_cost_fn('input_tokens', exclude_prover_calls=True)),
        output_tokens=int(get_total_cost_fn('output_tokens', exclude_prover_calls=True)),
    )


def _extract_proof_data(formalizations, load_code: bool = False) -> Dict[str, List[AttemptPair]]:
    """Extract proof data from formalizations, grouped by model name.

    Returns:
        {model_name: [AttemptPair, ...]}
    """
    # Collect all attempts across formalizations
    all_attempts = []
    for form in formalizations:
        for attempt in form.proof_attempts:
            all_attempts.append(attempt)

    if not all_attempts:
        return {}

    # Group by model name
    by_model: Dict[str, list] = defaultdict(list)
    for attempt in all_attempts:
        model_name = extract_model_name(attempt.model_config_path)
        if model_name is None:
            model_name = "unknown"
        by_model[model_name].append(attempt)

    # For each model, group by initial_attempt_index to pair initials with corrections
    result: Dict[str, List[AttemptPair]] = {}
    for model_name, attempts in by_model.items():
        pairs = _pair_initials_with_corrections(attempts, load_code=load_code)
        if pairs:
            result[model_name] = pairs

    return result


def _pair_initials_with_corrections(attempts: list, load_code: bool = False) -> List[AttemptPair]:
    """Pair initial attempts with their corrections.

    Groups by initial_attempt_index (or attempt_id for initials).
    Corrections have correction_round_id > 0 and initial_attempt_index
    pointing to the original attempt's attempt_id.
    """
    # Separate initials and corrections
    initials = []
    corrections_by_initial: Dict[int, list] = defaultdict(list)

    for attempt in attempts:
        if attempt.correction_round_id == 0:
            initials.append(attempt)
        else:
            # initial_attempt_index links correction to its original
            iai = attempt.initial_attempt_index
            if iai is not None:
                corrections_by_initial[iai].append(attempt)

    # Build pairs
    pairs = []
    for initial in initials:
        corrections = corrections_by_initial.get(initial.attempt_id, [])
        corrections.sort(key=lambda a: a.correction_round_id)

        pair = AttemptPair(
            initial=_proof_attempt_to_attempt_data(initial, load_code=load_code),
            corrections=[_proof_attempt_to_attempt_data(c, load_code=load_code) for c in corrections],
        )
        pairs.append(pair)

    return pairs


def _proof_attempt_to_attempt_data(attempt, load_code: bool = False) -> AttemptData:
    """Convert a ProofAttempt to AttemptData."""
    dc = attempt.detailed_cost or {}
    errors = attempt.compilation_result.errors if attempt.compilation_result else []
    error_messages = [
        err.get("data", "").strip() if isinstance(err, dict) else str(err).strip()
        for err in errors
        if (isinstance(err, dict) and err.get("data")) or (isinstance(err, str) and err.strip())
    ] or None
    return AttemptData(
        success=attempt.is_passing(),
        cost=DetailedCost(
            input_sflops=int(dc.get('input_sflops', 0)),
            output_sflops=int(dc.get('output_sflops', 0)),
            input_tokens=int(dc.get('input_tokens', 0)),
            output_tokens=int(dc.get('output_tokens', 0)),
        ),
        proof_length=attempt.reasoning_summary.get('proof_length', 0) if attempt.reasoning_summary else None,
        num_errors=len(errors) if errors else (attempt.compilation_summary.get('total_errors') if attempt.compilation_summary else None),
        used_lemma_ids=attempt.used_lemma_ids.copy() if attempt.used_lemma_ids else None,
        correction_round_id=attempt.correction_round_id,
        attempt_id=attempt.attempt_id,
        code=attempt.code if load_code else None,
        error_messages=error_messages,
    )
