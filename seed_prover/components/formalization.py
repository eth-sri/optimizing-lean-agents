import os
import json
import re
from typing import Dict, List, Any
from loguru import logger
from jload import jsave
from core.component import Component
from prover.query.api import APIQuery
from lean_compiler.repl_scheduler import scheduler
from utils import problem_check, extract_code
from metadata_utils import add_lemma, generate_uid, copy_metadata, add_formalization
import pdb

def handle(text):
    """Filter out import, set_option, and open statements from Lean code."""
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not (
            line.strip().startswith('import') or
            line.strip().startswith('set_option') or
            line.strip().startswith('open')
    )]
    return '\n'.join(filtered_lines)


def remove_sample_suffix(problem_id: str) -> str:
    """
    Remove the _sample_{i} suffix from a problem ID.

    Args:
        problem_id: The problem ID that may contain _sample_{i} suffix

    Returns:
        Problem ID with _sample_{i} suffix removed (if present)
    """
    # Remove _sample_{i} suffix pattern
    return re.sub(r'_sample_\d+$', '', problem_id)


def enforce_lemma_name(code: str, lemma_id: int, metadata: dict = None, formalization_id: int = None) -> str:
    """
    Enforce correct lemma naming in Lean code with unique prefix.

    Replaces 'theorem <anything>' with 'theorem {parent_problem_id}_lemma{lemma_id}'
    or 'theorem {parent_problem_id}_lemma{lemma_id}_f{formalization_id}' if formalization_id is provided.
    This ensures consistent naming and avoids collisions across rounds, breakdowns, and formalizations.

    Args:
        code: The Lean code from LLM
        lemma_id: The numeric ID for this lemma
        metadata: Metadata dict containing parent_problem_id (and optionally origin_problem_id)
        formalization_id: Optional formalization ID to append to the name (for multi-formalization support)

    Returns:
        Code with corrected lemma name using unique prefix
    """
    # Determine the prefix from metadata
    if metadata:
        # Use parent_problem_id if available, otherwise fall back to origin_problem_id
        parent_id = metadata.get('parent_problem_id') or metadata.get('origin_problem_id', '')
        if parent_id:
            if lemma_id == -1:
                lemma_name = f'{parent_id}'
            else:
                lemma_name = f'{parent_id}_lemma{lemma_id}'
                # Append formalization_id if provided (for multi-formalization support)
                if formalization_id is not None:
                    lemma_name += f'_f{formalization_id}'
        else:
            # Fallback to old behavior if no metadata
            lemma_name = f'lemma{lemma_id}'
            if formalization_id is not None:
                lemma_name += f'_f{formalization_id}'
    else:
        # Fallback to old behavior if no metadata
        lemma_name = f'lemma{lemma_id}'
        if formalization_id is not None:
            lemma_name += f'_f{formalization_id}'

    # Pattern to match: theorem <name> (where name can be any valid identifier)
    pattern = r'\btheorem\s+(\w+)'
    replacement = f'theorem {lemma_name}'

    corrected = re.sub(pattern, replacement, code, count=1)  # Only replace first occurrence

    return corrected


class FormalizationComponent(Component):
    """
    Component to formalize parsed breakdowns into Lean statements.

    Takes a parsed_breakdown with lemmas and theorem, and can either:
    1. Formalize lemmas by sampling from an LLM (when sample_formalizations=True)
    2. Use formal_statement from parsed_breakdown (when sample_formalizations=False)

    The theorem's formal_statement is always taken directly from the dataset.
    """
    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)

        self.sample_formalizations = self.config.get('sample_formalizations', True)

        self.base_template_path = f"seed_prover/template/formalization/{self.config.get('template', 'seed_prover')}.md"
        self.base_template = self._load_template(self.base_template_path)

        # Initialize validation configuration
        self.validation_config = self.config.get('validation_config', {"enabled": False})
        self.validation_enabled = self.validation_config.get('enabled', False)
        self.validation_querier = None
        self.validation_template = None
        self.validation_type = self.validation_config.get('type', 'binary')

    def load_model(self):
                # Only initialize the LLM querier if we're sampling formalizations
        self.querier = None
        if self.sample_formalizations:
            model = self.model_config.get('model')
            api = self.model_config.get('api')
            max_tokens = self.model_config.get('max_tokens', 40960)
            temperature = self.model_config.get('temperature', 1.0)

            kwargs = self.model_config.copy()
            del kwargs["model"]
            del kwargs["api"]
            if 'date' in kwargs:
                del kwargs['date']
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

            # Get verbosity from global config
            verbosity = self.global_config.get('verbosity', 3)

            self.querier = APIQuery(
                model=model,
                api=api,
                verbosity=verbosity,
                **kwargs
            )
        else:
            # Skip model loading when not sampling formalizations
            verbosity = self.global_config.get('verbosity', 3)
            if verbosity >= 1:
                logger.info("Skipping LLM model initialization (sample_formalizations=False)")

        if self.validation_enabled:
            # Load validation template based on type
            if self.validation_type == 'select':
                validation_template_path = self.validation_config.get(
                    'select_template',
                    'seed_prover/template/formalization/select_formalization.md'
                )
            else:  # binary (default)
                validation_template_path = self.validation_config.get(
                    'binary_template',
                    self.validation_config.get('template', 'seed_prover/template/formalization/validate_formalization.md')
                )
            self.validation_template = self._load_template(validation_template_path)

            # Initialize validation querier with separate model config
            validation_model_config = self.validation_config.get('model_config', {})
            if isinstance(validation_model_config, str):
                # If it's a path to a config file, load it
                import yaml
                with open(validation_model_config, 'r') as f:
                    validation_model_config = yaml.safe_load(f)

            model = validation_model_config.get('model')
            api = validation_model_config.get('api')
            max_tokens = validation_model_config.get('max_tokens', 4096)
            temperature = validation_model_config.get('temperature', 0.0)

            kwargs = validation_model_config.copy()
            if 'model' in kwargs:
                del kwargs['model']
            if 'api' in kwargs:
                del kwargs['api']
            if 'date' in kwargs:
                del kwargs['date']
            kwargs['max_tokens'] = max_tokens
            kwargs['temperature'] = temperature

            verbosity = self.global_config.get('verbosity', 3)

            self.validation_querier = APIQuery(
                model=model,
                api=api,
                verbosity=verbosity,
                **kwargs
            )

            if verbosity >= 1:
                logger.info(f"Validation enabled with model: {model}, type: {self.validation_type}")

    def unload_model(self):
        if hasattr(self, 'querier') and self.querier is not None:
            if hasattr(self.querier, 'cleanup'):
                self.querier.cleanup()
        if hasattr(self, 'validation_querier') and self.validation_querier is not None:
            if hasattr(self.validation_querier, 'cleanup'):
                self.validation_querier.cleanup()

    def _load_template(self, template_path):
        if not template_path or not os.path.exists(template_path):
            raise ValueError(f"Template path {template_path} does not exist or is not set in config")

        with open(template_path, "r") as f:
            template = f.read()

        return template

    def _validate_formalizations(self, validation_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate compiled formalizations using a separate LLM.

        Args:
            validation_items: List of dicts containing:
                - 'problem_id': Problem identifier
                - 'lemma_id': Lemma identifier
                - 'sample_idx': Sample index (optional, for sampled formalizations)
                - 'statement': Informal statement
                - 'assumption': Informal assumptions
                - 'proof': Informal proof idea
                - 'formal_statement': The compiled Lean code to validate

        Returns:
            List of dicts with validation results:
                - 'problem_id', 'lemma_id', 'sample_idx'
                - 'verdict': 'yes' or 'no'
                - 'raw_response': Full LLM response
                - 'validation_prompt': The prompt sent
                - 'cost': Query cost info
        """
        if not self.validation_enabled or not validation_items:
            return []

        import re
        import time

        verbosity = self.global_config.get('verbosity', 3)

        # Prepare validation prompts
        validation_prompts = []
        for item in validation_items:
            prompt = self.validation_template.format(
                original_statement=item.get('original_statement', ''),
                statement=item.get('statement', ''),
                assumption=item.get('assumption', ''),
                proof=item.get('proof', ''),
                compiled_code=item.get('compiled_code', '')
            )
            # Format as chat messages for API
            validation_prompts.append([
                {
                    "role": "user",
                    "content": prompt
                }
            ])

        if verbosity >= 2:
            logger.info(f"Validating {len(validation_prompts)} formalizations...")

        # Run validation queries in parallel
        results = []
        total_cost = 0
        start_time = time.time()

        for i, response, detailed_cost in self.validation_querier.run_queries(validation_prompts):
            # Extract reasoning/thinking from response (before stripping)
            # The model outputs thinking that ends with </think>, then the actual content
            if '</think>' in response:
                reasoning, response_content = response.split('</think>', 1)  # Split only on first occurrence
            else:
                reasoning = ""
                response_content = response

            # Parse verdict from response (after stripping thinking)
            verdict_match = re.search(r'<verdict>(yes|no)</verdict>', response_content, re.IGNORECASE)
            verdict = verdict_match.group(1).lower() if verdict_match else 'unknown'

            # Extract metadata for this validation item
            item_metadata = validation_items[i].get('metadata', {})

            results.append({
                'problem_id': validation_items[i].get('problem_id'),
                'lemma_id': validation_items[i].get('lemma_id'),
                'sample_idx': validation_items[i].get('sample_idx'),
                'item_idx': validation_items[i].get('item_idx'),  # Include item_idx for matching
                'name': validation_items[i].get('name'),  # Required for non-sampled mode matching
                'verdict': verdict,
                'reasoning': reasoning,  # Include extracted reasoning
                'raw_response': response,
                'validation_prompt': validation_prompts[i][0]['content'],  # Extract content from message format
                # Lemma details for analysis
                'statement': validation_items[i].get('statement', ''),
                'assumption': validation_items[i].get('assumption', ''),
                'proof': validation_items[i].get('proof', ''),
                'compiled_code': validation_items[i].get('compiled_code', ''),
                'cost': detailed_cost,
                'timestamp': time.time(),
                # Metadata for traceability
                'metadata': item_metadata,
                'uid': generate_uid(item_metadata) if item_metadata else None
            })
            total_cost += detailed_cost.get('total_cost', 0)

        elapsed = time.time() - start_time

        # Count verdicts
        yes_count = sum(1 for r in results if r['verdict'] == 'yes')
        no_count = sum(1 for r in results if r['verdict'] == 'no')
        unknown_count = sum(1 for r in results if r['verdict'] == 'unknown')

        if verbosity >= 1:
            logger.info(
                f"Validation complete: {yes_count} yes, {no_count} no, {unknown_count} unknown "
                f"(${total_cost:.4f}, {elapsed:.1f}s)"
            )

        return results

    def _select_best_formalization(self, selection_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Select the best formalization from multiple compiled samples using LLM.

        Args:
            selection_items: List of dicts, each containing:
                - 'item_idx': Index in data_list
                - 'lemma_id': Lemma identifier
                - 'original_statement': Original problem statement
                - 'statement': Lemma statement
                - 'assumption': Lemma assumptions
                - 'proof': Lemma proof idea
                - 'samples': List of compiled samples with 'sample_idx' and 'compiled_code'

        Returns:
            List of dicts with selection results:
                - 'item_idx', 'lemma_id'
                - 'selected_sample_idx': Index of selected sample
                - 'raw_response': Full LLM response
                - 'cost': Query cost info
        """
        if not self.validation_enabled or not selection_items:
            return []

        import re
        import time

        verbosity = self.global_config.get('verbosity', 3)

        # Prepare selection prompts
        selection_prompts = []
        for item in selection_items:
            # Concatenate all formalization codes with indices (zero-based)
            lemma_code = '\n\n'.join([f"formalization {s['sample_idx']}:\n{s['compiled_code']}" for s in item['samples']])

            prompt = self.validation_template.format(
                original_statement=item.get('original_statement', ''),
                statement=item.get('statement', ''),
                assumption=item.get('assumption', ''),
                proof=item.get('proof', ''),
                lemma_code=lemma_code,
                i=item.get('lemma_id', '')
            )
            selection_prompts.append([
                {
                    "role": "user",
                    "content": prompt
                }
            ])

        if verbosity >= 2:
            logger.info(f"Selecting best formalization for {len(selection_prompts)} lemmas...")

        # Run selection queries in parallel
        results = []
        total_cost = 0
        start_time = time.time()

        for i, response, detailed_cost in self.validation_querier.run_queries(selection_prompts):
            # Extract thinking from response (before stripping)
            if '</think>' in response:
                reasoning, response_content = response.split('</think>', 1)
            else:
                reasoning = ""
                response_content = response

            # Parse the selected index from response
            # Look for code block containing just an integer (zero-based index)
            index_match = re.search(r'```\s*(\d+)\s*```', response_content)

            # If no code block found, try to find a bare number at the end
            if not index_match:
                index_match = re.search(r'(\d+)\s*$', response_content.strip())

            item = selection_items[i]
            selected_sample_idx = None
            selected_code = ""

            if index_match:
                # Extract the index from the response
                selected_idx = int(index_match.group(1).strip())

                # Validate that this index exists in the samples
                valid_indices = [s['sample_idx'] for s in item['samples']]
                if selected_idx in valid_indices:
                    selected_sample_idx = selected_idx
                    # Get the actual code for this index
                    for sample in item['samples']:
                        if sample['sample_idx'] == selected_idx:
                            selected_code = sample['compiled_code']
                            break
                    if verbosity >= 2:
                        logger.info(f"Selected formalization {selected_idx} for item {item.get('item_idx')}, lemma {item.get('lemma_id')}")
                else:
                    if verbosity >= 1:
                        logger.warning(f"Invalid index {selected_idx} for item {item.get('item_idx')}, lemma {item.get('lemma_id')}, using first sample")
                    selected_sample_idx = item['samples'][0]['sample_idx'] if item['samples'] else None
                    selected_code = item['samples'][0]['compiled_code'] if item['samples'] else ""
            else:
                if verbosity >= 1:
                    logger.warning(f"Could not parse index from response for item {item.get('item_idx')}, lemma {item.get('lemma_id')}, using first sample")
                selected_sample_idx = item['samples'][0]['sample_idx'] if item['samples'] else None
                selected_code = item['samples'][0]['compiled_code'] if item['samples'] else ""

            results.append({
                'item_idx': item.get('item_idx'),
                'lemma_id': item.get('lemma_id'),
                'selected_sample_idx': selected_sample_idx,
                'selected_code': selected_code,
                'reasoning': reasoning,
                'raw_response': response,
                'cost': detailed_cost,
                'timestamp': time.time()
            })
            total_cost += detailed_cost.get('total_cost', 0)

        elapsed = time.time() - start_time

        if verbosity >= 1:
            logger.info(
                f"Selection complete: {len(results)} lemmas processed "
                f"(${total_cost:.4f}, {elapsed:.1f}s)"
            )

        return results

    def _prepare_prompts_for_item(self, parsed_breakdown: Dict[str, Any], informal_breakdown: str) -> List[Dict[str, Any]]:
        """
        Prepare prompts for all lemmas in a parsed breakdown.

        Note: We only formalize lemmas. The theorem's formal_statement comes from the dataset.

        Args:
            parsed_breakdown: Dict with 'lemmas' list and 'theorem' dict
            informal_breakdown: The original informal breakdown text

        Returns:
            List of dicts with 'type' (lemma), 'id', and 'prompt'
        """
        prompts = []

        # Process lemmas only
        lemmas = parsed_breakdown.get("lemmas", [])
        for lemma in lemmas:
            if lemma is None:
                continue
            statement = lemma.get("statement", "")
            assumption = lemma.get("assumption", "")
            proof = lemma.get("proof", "")
            lemma_id = lemma.get("id", "")

            # Format the template
            prompt = self.base_template.format(
                informal_breakdown=informal_breakdown,
                proof_type="lemma",
                statement=f"Statement: {statement}",
                assumption=f"Assumption: {assumption}",
                proof=f"Proof idea: {proof}",
                i = lemma_id
            )

            prompts.append({
                "type": "lemma",
                "id": lemma_id,
                "prompt": prompt
            })

        # Note: Theorem is NOT formalized here - it comes from dataset's formal_statement

        return prompts

    def _compile(self, data_list):
        """
        Compile all formal statements to verify they are syntactically correct.

        Args:
            data_list: List of items with formal statements

        Returns:
            List of compilation results
        """
        codes = []
        for item in data_list:
            parsed_breakdown = item.get("parsed_breakdown", {})

            # Compile each lemma
            for lemma in parsed_breakdown.get("lemmas", []):
                if lemma is None:
                    continue
                formal_statement = lemma.get("formal_statement_raw", "")
                if not formal_statement or formal_statement == "None":
                    continue

                extracted_code = extract_code(formal_statement)
                if extracted_code == "None" or extracted_code is None:
                    continue

                code = handle(extracted_code).strip()
                codes.append({
                    "name": f"{item.get('problem_id', 'unknown')}_lemma_{lemma.get('id')}",
                    "code": code,
                    "problem_id": item.get("problem_id", "unknown"),
                    "lemma_id": lemma.get("id"),
                    "type": "lemma"
                })

            # Compile theorem
            theorem = parsed_breakdown.get("theorem", {})
            if theorem:
                formal_statement = theorem.get("formal_statement_raw", "")
                if formal_statement and formal_statement != "None":
                    extracted_code = extract_code(formal_statement)
                    if extracted_code != "None" and extracted_code is not None:
                        code = handle(extracted_code)
                        codes.append({
                            "name": f"{item.get('problem_id', 'unknown')}_theorem",
                            "code": code,
                            "problem_id": item.get("problem_id", "unknown"),
                            "type": "theorem"
                        })

        if not codes:
            logger.warning("No code to compile")
            return []

        outputs = scheduler(codes, num_workers=self.config.get("cpu", 4))

        return outputs

    def _check_pass(self, compilation_result):
        """
        Filter compilation results to only those that passed.

        Args:
            compilation_result: List of compilation results

        Returns:
            List of passing compilation results
        """
        pass_compilation_results = []

        for repl_output in compilation_result:
            if "errors" not in repl_output.get("compilation_result", {}):
                continue

            is_pass = repl_output.get("compilation_result").get("pass")

            if is_pass:
                pass_compilation_results.append(repl_output)

        return pass_compilation_results

    def _update_with_compilation_results(self, data_list, compilation_results):
        """
        Update data_list with compilation results.

        Args:
            data_list: List of items
            compilation_results: List of compilation results

        Returns:
            Updated data_list
        """
        # Create lookup map
        comp_map = {}
        for comp in compilation_results:
            name = comp.get("name", "")
            comp_map[name] = comp

        # Update each item
        for item in data_list:
            problem_id = item.get("problem_id", "unknown")
            parsed_breakdown = item.get("parsed_breakdown", {})

            # Update lemmas
            for lemma in parsed_breakdown.get("lemmas", []):
                if lemma is None:
                    continue
                lemma_name = f"{problem_id}_lemma_{lemma.get('id')}"
                if lemma_name in comp_map:
                    comp_result = comp_map[lemma_name]
                    lemma["compilation_result"] = comp_result.get("compilation_result", {})
                    lemma["compiled_code"] = comp_result.get("code", "")
                    lemma["compilation_pass"] = comp_result.get("compilation_result", {}).get("pass", False)

            # Update theorem
            theorem = parsed_breakdown.get("theorem", {})
            if theorem:
                theorem_name = f"{problem_id}_theorem"
                if theorem_name in comp_map:
                    comp_result = comp_map[theorem_name]
                    theorem["compilation_result"] = comp_result.get("compilation_result", {})
                    theorem["compiled_code"] = comp_result.get("code", "")
                    theorem["compilation_pass"] = comp_result.get("compilation_result", {}).get("pass", False)

            item["parsed_breakdown"] = parsed_breakdown

        return data_list

    def _cleanup_model(self):
        """
        Cleanup VLLM model to free GPU memory.
        """
        if self.querier is not None and hasattr(self.querier, 'vllm_model'):
            verbosity = self.global_config.get('verbosity', 3)
            if verbosity >= 1:
                logger.info("Releasing VLLM model to free GPU memory...")
            import gc
            import torch

            # Delete the VLLM model
            del self.querier.vllm_model
            del self.querier

            # Force garbage collection
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            if verbosity >= 1:
                logger.info("VLLM model released successfully")

    def _save_lemma_problems(self, data_list, round_num=0):
        """
        Extract lemmas and theorem from parsed_breakdown and save them as separate problems.
        Similar to delta_prover's _save_lemma_problems.
        """
        new_lemma_problems = []

        for item in data_list:
            parsed_breakdown = item.get("parsed_breakdown", {})

            # Process lemmas that compiled successfully
            lemmas = parsed_breakdown.get("selected_formalizations", [])

            # Get parent metadata for generating UIDs
            parent_metadata = item.get("metadata")
            if not parent_metadata:
                raise ValueError(f"Item missing metadata field: {item}")

            # Build a map of lemma_id (integer) to compiled code for dependency resolution
            # Dependencies are now stored as integers, context is inferred from metadata
            lemma_code_map = {}
            for lemma in lemmas:
                if lemma is None:
                    continue
                if lemma.get("compilation_pass", False):
                    lemma_id = lemma.get("id")
                    formalization_id = lemma.get("formalization_id")
                    compiled_code = lemma.get("compiled_code", "")
                    if compiled_code:
                        # Store as axiom for use in dependencies (keyed by integer lemma_id)
                        axiom_code = compiled_code.replace("theorem", "axiom", 1).split(':= by')[0].strip()
                        lemma_code_map[(lemma_id, formalization_id)] = axiom_code

            for lemma in lemmas:
                if lemma is None:
                    continue

                lemma_id = lemma.get("id")

                if not lemma.get("compilation_pass", False):
                    continue

                formal_statement = lemma.get("formal_statement", "")
                compiled_code = lemma.get("compiled_code", "")
                formalization_id = lemma.get("formalization_id")

                if formal_statement and compiled_code:
                    # Extract the actual lemma name from already-compiled code
                    # (name was already enforced during compilation)
                    lemma_name_match = re.search(r'\btheorem\s+(\S+)', compiled_code)
                    lemma_name = lemma_name_match.group(1) if lemma_name_match else None

                    # Get dependencies for this lemma
                    lemma_dependencies = lemma.get("dependencies", [])
                    dependency_axioms = []
                    for dep_id in lemma_dependencies:
                        for dep_form_id in range(self.config.get('sample_times', 1)):
                            if (dep_id, dep_form_id) in lemma_code_map:
                                dependency_axioms.append(lemma_code_map[(dep_id, dep_form_id)])

                    # Build the formal statement with dependency axioms
                    if dependency_axioms:
                        dependency_str = '\n'.join([f'-- Lemma dependency:\n{axiom}' for axiom in dependency_axioms])
                        lemma_formal_statement = f"""-- Dependent lemmas (written as axioms because they are assumed to be true):
{dependency_str}

{compiled_code}
"""
                    else:
                        lemma_formal_statement = f"{compiled_code}\n"

                    # Build informal prefix with proof idea
                    lemma_informal_prefix = f"Proof idea: {lemma.get('proof', '')}"

                    # Get metadata from parent item and add lemma_id
                    parent_metadata = item.get("metadata")
                    if not parent_metadata:
                        raise ValueError(f"Item missing metadata field: {item}")

                    lemma_metadata = add_lemma(parent_metadata, lemma_id)

                    # Add formalization_id
                    if formalization_id is not None:
                        lemma_metadata = add_formalization(lemma_metadata, formalization_id)

                    lemma_problem = {
                        "informal_prefix": lemma_informal_prefix,
                        "formal_statement": lemma_formal_statement,
                        "lean4_code": lemma_formal_statement,
                        "metadata": lemma_metadata,
                        "uid": generate_uid(lemma_metadata),
                        "type": "lemma",
                        "lemma_name": lemma_name  # For direct axiom name matching
                    }
                    new_lemma_problems.append(lemma_problem)

            # Process theorem if it compiled successfully
            theorem = parsed_breakdown.get("theorem")

            # Add validation to handle None theorem
            if theorem is not None:
                theorem_statement = enforce_lemma_name(handle(theorem.get("formal_statement", "")), -1, parent_metadata)

                # Get dependencies for the theorem (uses the same approach as lemmas)
                theorem_dependencies = theorem.get("dependencies", [])
                theorem_dependency_axioms = []
                for dep_id in theorem_dependencies:
                    for dep_form_id in range(self.config.get('sample_times', 1)):
                        if (dep_id, dep_form_id) in lemma_code_map:
                            theorem_dependency_axioms.append(lemma_code_map[(dep_id, dep_form_id)])

                # Build theorem formal statement with dependency axioms
                if theorem_dependency_axioms:
                    dependency_str = '\n\n'.join(theorem_dependency_axioms)
                    theorem_formal_statement = f"""{dependency_str}

{theorem_statement}
"""
                else:
                    theorem_formal_statement = f"{theorem_statement}\n"

                # Build informal prefix with theorem statement and proof idea
                theorem_informal_prefix = f"""Theorem statement: {theorem.get("statement")}

Proof idea: {theorem.get("proof")}"""

                # Get metadata from parent item and add lemma_id=-1 for theorem
                parent_metadata = item.get("metadata")
                if not parent_metadata:
                    raise ValueError(f"Item missing metadata field: {item}")

                theorem_metadata = add_lemma(parent_metadata, -1)

                theorem_problem = {
                    "informal_prefix": theorem_informal_prefix,
                    "formal_statement": theorem_formal_statement,
                    "lean4_code": theorem_formal_statement,
                    "metadata": theorem_metadata,
                    "uid": generate_uid(theorem_metadata),
                    "type": "theorem"
                }

                new_lemma_problems.append(theorem_problem)
            else:
                logger.warning(f"Skipping theorem for {item.get('uid', item.get('name', 'unknown'))}: theorem is None in parsed_breakdown")

        # Split lemmas and theorems into separate files
        lemmas_only = [p for p in new_lemma_problems if p.get("type") == "lemma"]
        theorems_only = [p for p in new_lemma_problems if p.get("type") == "theorem"]

        # Save to prover directory
        base_output_dir = self.global_config.get("output_dir")
        prover_dir = os.path.join(base_output_dir, f"round{round_num}", "prover")
        os.makedirs(prover_dir, exist_ok=True)

        theorems_file = os.path.join(prover_dir, "theorems.json")
        jsave(theorems_only, theorems_file)
        logger.info(f"Saved {len(theorems_only)} theorem problems to {theorems_file}")

        lemmas_file = os.path.join(prover_dir, "lemmas.json")
        jsave(lemmas_only, lemmas_file)
        logger.info(f"Saved {len(lemmas_only)} lemma problems to {lemmas_file}")

    def process(self, data_list: List[Dict[str, Any]], round_num: int = 0) -> List[Dict[str, Any]]:
        """
        Process a list of data items with parsed breakdowns.

        Args:
            data_list: List of dictionaries containing 'parsed_breakdown'
            round_num: Current processing round

        Returns:
            List of dictionaries with formalized statements added
        """
        verbosity = self.global_config.get('verbosity', 3)

        sample_times = self.config.get('sample_times', 1)
        keep_samples = self.config.get('keep_samples', 1)

        # VALIDATION: Check for double-sampling (can't sample in both breakdown_parser and here)
        if sample_times > 1:
            for item in data_list:
                problem_id = item.get("problem_id", "")
                # Check for breakdown_parser sampling pattern (_sample_\d+)
                if re.search(r'_sample_\d+$', problem_id):
                    raise ValueError(
                        f"Double-sampling detected: problem_id '{problem_id}' already has _sample_<digit> suffix "
                        f"from breakdown_parser sampling, but formalization has sample_times={sample_times}. "
                        f"Please ensure only one sampling stage is active (either breakdown_parser or formalization)."
                    )

        logger.info(f"Formalizing {len(data_list)} items with {sample_times} samples per lemma...")

        # Initialize validation results list (used if validation is enabled)
        all_validation_results = []
        all_selection_results = []  # For select mode

        # Check for breakdown_parser samples (items with _sample_{digit} suffix from breakdown_parser sample_times > 1)
        has_breakdown_parser_samples = any(
            re.search(r'_sample_\d+$', item.get("problem_id", "")) is not None
            for item in data_list
        )

        # Validate: breakdown_parser sampling requires sample_formalizations=False
        if has_breakdown_parser_samples and self.sample_formalizations:
            error_msg = (
                "ERROR: Detected breakdown_parser samples (from breakdown_parser sample_times > 1) "
                "but sample_formalizations=True. This creates conflicting sampling strategies. "
                "Please set sample_formalizations=False when using breakdown_parser sample_times > 1."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.sample_formalizations:
            logger.info(f"Formalizing {len(data_list)} items with {sample_times} samples per lemma (sampling from LLM)...")
        else:
            if has_breakdown_parser_samples:
                # Count unique origin problems
                unique_origins = len(set(item.get("origin_problem_id") for item in data_list if "origin_problem_id" in item))
                logger.info(f"Formalizing {len(data_list)} breakdown_parser samples ({unique_origins} unique problems) using formal_statement from parsed breakdown...")
            else:
                logger.info(f"Formalizing {len(data_list)} items using formal_statement from parsed breakdown...")

        # Branch based on sample_formalizations flag
        if self.sample_formalizations:
            # === SAMPLE FROM LLM ===
            # STEP 1: Prepare all prompts from all items at once (with sampling)
            all_prompts = []
            prompt_metadata = []  # Track which item, lemma, and sample each prompt belongs to

            for item_idx, item in enumerate(data_list):
                parsed_breakdown = item.get("parsed_breakdown", {})
                informal_breakdown = item.get("informal_breakdown", "")

                if not parsed_breakdown or "lemmas" not in parsed_breakdown:
                    if verbosity >= 1:
                        logger.warning(f"No valid parsed_breakdown found for item: {item.get('problem_id', 'unknown')}")
                    continue

                # Prepare prompts for lemmas only
                prompt_info_list = self._prepare_prompts_for_item(parsed_breakdown, informal_breakdown)

                # Generate sample_times samples for each lemma
                for prompt_info in prompt_info_list:
                    for sample_idx in range(sample_times):
                        all_prompts.append([{"role": "user", "content": prompt_info["prompt"]}])
                        prompt_metadata.append({
                            "item_idx": item_idx,
                            "lemma_id": prompt_info["id"],
                            "sample_idx": sample_idx,
                            "type": prompt_info["type"],
                            "prompt": prompt_info["prompt"]
                        })

            if not all_prompts:
                logger.warning("No prompts to process")
                return data_list

            logger.info(f"Running {len(all_prompts)} formalization queries in parallel ({sample_times} samples x {len(all_prompts)//sample_times} lemmas)...")

            # STEP 2: Run all queries in parallel
            results = []
            total_cost = 0
            for i, response, detailed_cost in self.querier.run_queries(all_prompts):
                results.append({
                    "index": i,
                    "response": response,
                    "cost": detailed_cost
                })
                total_cost += detailed_cost.get("total_cost", 0) if isinstance(detailed_cost, dict) else 0

            # STEP 3: Group results by (item_idx, lemma_id) and compile all samples
            from collections import defaultdict
            lemma_samples = defaultdict(list)  # (item_idx, lemma_id) -> list of samples

            for result in results:
                idx = result["index"]
                metadata = prompt_metadata[idx]
                item_idx = metadata["item_idx"]
                lemma_id = metadata["lemma_id"]
                sample_idx = metadata["sample_idx"]
                formal_statement_raw = result["response"]

                # Strip thinking from formalization response (if present)
                if '</think>' in formal_statement_raw:
                    formal_statement = formal_statement_raw.split('</think>', 1)[1].strip()
                else:
                    formal_statement = formal_statement_raw

                lemma_samples[(item_idx, lemma_id)].append({
                    "sample_idx": sample_idx,
                    "lemma_id": lemma_id,
                    "formal_statement": formal_statement,
                    "formal_statement_raw": formal_statement_raw,
                    "prompt": metadata.get("prompt", ""),
                    "detailed_cost": result["cost"]  # Add detailed_cost from querier
                })

            # STEP 4: Compile all samples and select first compiling one
            logger.info(f"Compiling {len(lemma_samples)} lemmas with {sample_times} samples each...")

            # Build compilation tasks for all samples
            compile_tasks = []
            # Map from name to metadata (since scheduler only preserves 'name' field)
            name_to_metadata = {}

            for (item_idx, lemma_id), samples in lemma_samples.items():
                item = data_list[item_idx]
                parent_metadata = item.get("metadata", {})

                for sample in samples:
                    formal_statement = sample["formal_statement"]
                    extracted_code = extract_code(formal_statement)

                    # Enforce correct lemma naming with formalization ID
                    formal_statement = enforce_lemma_name(extracted_code, lemma_id, parent_metadata, sample["sample_idx"])

                    if verbosity >= 3 and formal_statement != extracted_code:
                        logger.debug(f"Corrected lemma name for item {item_idx}, lemma{lemma_id}, sample {sample['sample_idx']}")

                    if formal_statement and formal_statement != "None":
                        code = handle(formal_statement).strip()

                        # Create metadata for this formalization sample
                        sample_metadata = add_lemma(parent_metadata, lemma_id)
                        sample_metadata = add_formalization(sample_metadata, sample["sample_idx"])
                        name = generate_uid(sample_metadata)

                        compile_tasks.append({
                            "name": name,
                            "code": code,
                        })
                        # Store metadata separately for retrieval after compilation
                        name_to_metadata[name] = {
                            "lemma_id": lemma_id,
                            "sample_idx": sample["sample_idx"],
                            "item_idx": item_idx,
                            "type": "lemma_sample",
                            "metadata": sample_metadata
                        }

            if compile_tasks:
                sample_compilation_results = scheduler(compile_tasks, num_workers=self.config.get("cpu", 4))
            else:
                sample_compilation_results = []

            # Group compilation results by (item_idx, lemma_id)
            from collections import defaultdict
            compilation_by_lemma = defaultdict(list)
            for comp_result in sample_compilation_results:
                name = comp_result.get("name", "")
                metadata = name_to_metadata.get(name, {})

                item_idx = metadata.get("item_idx")
                lemma_id = metadata.get("lemma_id")
                sample_idx = metadata.get("sample_idx")
                is_pass = comp_result.get("compilation_result", {}).get("pass", False)

                compilation_by_lemma[(item_idx, lemma_id)].append({
                    "sample_idx": sample_idx,
                    "pass": is_pass,
                    "compilation_result": comp_result.get("compilation_result", {}),
                    "compiled_code": comp_result.get("code", "")
                })

        if self.sample_formalizations:
            # STEP 4.5: Validate compiled samples (if validation enabled)
            from collections import defaultdict
            validation_by_lemma = defaultdict(dict)  # {(item_idx, lemma_id): {sample_idx: validation_result}}
            selection_by_lemma = {}  # {(item_idx, lemma_id): selected_sample_idx} for select mode

            if self.validation_enabled:
                if self.validation_type == 'select':
                    # SELECT MODE: Group samples by lemma and select the best one
                    # Prepare selection items grouped by (item_idx, lemma_id)
                    selection_items_by_lemma = defaultdict(lambda: {'samples': []})

                    for comp_result in sample_compilation_results:
                        name = comp_result.get("name", "")
                        metadata = name_to_metadata.get(name, {})
                        item_idx = metadata.get("item_idx")
                        lemma_id = metadata.get("lemma_id")
                        sample_idx = metadata.get("sample_idx")
                        is_pass = comp_result.get("compilation_result", {}).get("pass", False)

                        # Only include samples that compiled successfully
                        if is_pass:
                            item = data_list[item_idx]
                            parsed_breakdown = item.get("parsed_breakdown", {})
                            lemmas = parsed_breakdown.get("lemmas", [])

                            # Find the corresponding lemma data
                            lemma_data = None
                            for lemma in lemmas:
                                if lemma and lemma.get("id") == lemma_id:
                                    lemma_data = lemma
                                    break

                            if lemma_data:
                                key = (item_idx, lemma_id)
                                if 'item_idx' not in selection_items_by_lemma[key]:
                                    selection_items_by_lemma[key].update({
                                        'item_idx': item_idx,
                                        'lemma_id': lemma_id,
                                        'original_statement': item.get("informal_prefix", ""),
                                        'statement': lemma_data.get("statement", ""),
                                        'assumption': lemma_data.get("assumptions", ""),
                                        'proof': lemma_data.get("proof", "")
                                    })

                                selection_items_by_lemma[key]['samples'].append({
                                    'sample_idx': sample_idx,
                                    'compiled_code': comp_result.get("code", "")
                                })

                    # Filter out lemmas with no compiled samples
                    selection_items = [v for v in selection_items_by_lemma.values() if v['samples']]

                    # Run selection
                    selection_results = self._select_best_formalization(selection_items)
                    all_selection_results.extend(selection_results)

                    # Index selection results by (item_idx, lemma_id)
                    for s_result in selection_results:
                        item_idx = s_result.get('item_idx')
                        lemma_id = s_result.get('lemma_id')
                        selected_idx = s_result.get('selected_sample_idx')
                        if item_idx is not None and lemma_id is not None:
                            selection_by_lemma[(item_idx, lemma_id)] = selected_idx

                else:
                    # BINARY MODE: Validate each sample individually (existing logic)
                    # Prepare validation items for all compiled samples
                    validation_items = []
                    for comp_result in sample_compilation_results:
                        name = comp_result.get("name", "")
                        metadata = name_to_metadata.get(name, {})
                        item_idx = metadata.get("item_idx")
                        lemma_id = metadata.get("lemma_id")
                        sample_idx = metadata.get("sample_idx")
                        is_pass = comp_result.get("compilation_result", {}).get("pass", False)

                        # Only validate samples that compiled successfully
                        if is_pass:
                            item = data_list[item_idx]
                            parsed_breakdown = item.get("parsed_breakdown", {})
                            lemmas = parsed_breakdown.get("lemmas", [])

                            # Find the corresponding lemma data
                            lemma_data = None
                            for lemma in lemmas:
                                if lemma and lemma.get("id") == lemma_id:
                                    lemma_data = lemma
                                    break

                            if lemma_data:
                                # Get the sample's formal statement
                                sample_formal_statement = None
                                for sample in lemma_samples[(item_idx, lemma_id)]:
                                    if sample["sample_idx"] == sample_idx:
                                        sample_formal_statement = sample["formal_statement"]
                                        break

                                # Get the compiled code from the compilation result
                                compiled_code = comp_result.get("code", "")

                                # Build full metadata for this specific formalization
                                parent_metadata = item.get("metadata", {})
                                lemma_metadata = add_lemma(parent_metadata, lemma_id)
                                lemma_metadata = add_formalization(lemma_metadata, sample_idx)

                                validation_items.append({
                                    'problem_id': item.get("problem_id", "unknown"),
                                    'lemma_id': lemma_id,
                                    'sample_idx': sample_idx,
                                    'statement': lemma_data.get("statement", ""),
                                    'assumption': lemma_data.get("assumptions", ""),
                                    'proof': lemma_data.get("proof", ""),
                                    'formal_statement': sample_formal_statement or "",
                                    'compiled_code': compiled_code,
                                    'item_idx': item_idx,
                                    'original_statement': item.get("informal_prefix", ""),
                                    'metadata': lemma_metadata
                                })

                    # Run validation
                    validation_results = self._validate_formalizations(validation_items)
                    all_validation_results.extend(validation_results)

                    # Index validation results by (item_idx, lemma_id, sample_idx)
                    for v_result in validation_results:
                        item_idx = v_result.get('item_idx')
                        lemma_id = v_result.get('lemma_id')
                        sample_idx = v_result.get('sample_idx')
                        if item_idx is not None and lemma_id is not None and sample_idx is not None:
                            validation_by_lemma[(item_idx, lemma_id)][sample_idx] = v_result

            # STEP 5: Select top-K compiling AND validated samples for each lemma
            from collections import defaultdict

            selected_count = 0
            failed_count = 0

            # Group selected samples by item_idx
            selected_by_item = defaultdict(list)

            for (item_idx, lemma_id), samples in lemma_samples.items():
                # Get compilation results for this lemma
                compilations = compilation_by_lemma.get((item_idx, lemma_id), [])
                compilations.sort(key=lambda x: x["sample_idx"])  # Sort by sample index

                # Get validation results for this lemma (if validation enabled)
                validations = validation_by_lemma.get((item_idx, lemma_id), {})

                # Find top keep_samples samples that compiled AND validated (if validation enabled)
                lemma_selected = []  # Reset per lemma
                kept_samples = 0

                # For select mode, only keep 1 sample (the LLM-selected one)
                effective_keep_samples = 1 if (self.validation_enabled and self.validation_type == 'select') else keep_samples

                for i, compilation in enumerate(compilations):
                    sample_idx = compilation["sample_idx"]

                    # Check compilation
                    if not compilation["pass"]:
                        continue

                    # Check validation (if enabled)
                    if self.validation_enabled:
                        if self.validation_type == 'select':
                            # SELECT MODE: Only keep the LLM-selected sample
                            selected_idx = selection_by_lemma.get((item_idx, lemma_id))
                            if sample_idx == selected_idx:
                                lemma_selected.append({
                                    "sample": samples[sample_idx],
                                    "compilation": compilation,
                                    "validation": None,  # No individual validation in select mode
                                    "sample_idx": sample_idx,
                                    "lemma_id": lemma_id,
                                    "item_idx": item_idx
                                })
                                if verbosity >= 2:
                                    logger.info(f"Selected sample {sample_idx}/{sample_times} for item {item_idx}, lemma{lemma_id} (LLM selected)")
                                break  # Only keep one in select mode
                        else:
                            # BINARY MODE: Check validation verdict
                            validation = validations.get(sample_idx)
                            if validation and validation.get("verdict") == "yes":
                                 # Stop if we've collected enough samples
                                if kept_samples < effective_keep_samples:
                                    kept_samples += 1
                                    # Found a sample that both compiled and validated
                                    lemma_selected.append({
                                        "sample": samples[sample_idx],
                                        "compilation": compilation,
                                        "validation": validation,
                                        "sample_idx": sample_idx,
                                        "lemma_id": lemma_id,
                                        "item_idx": item_idx
                                    })

                                    if verbosity >= 2:
                                        logger.info(f"Selected sample {sample_idx}/{sample_times} for item {item_idx}, lemma{lemma_id} (compiled + validated)")
                                else:
                                    break
                    else:
                        if kept_samples < effective_keep_samples:
                            kept_samples += 1
                            # Validation not enabled, just check compilation
                            lemma_selected.append({
                                "sample": samples[sample_idx],
                                "compilation": compilation,
                                "validation": None,
                                "sample_idx": sample_idx,
                                "lemma_id": lemma_id,
                                "item_idx": item_idx
                            })
                            if verbosity >= 2:
                                logger.info(f"Selected sample {sample_idx}/{sample_times} for item {item_idx}, lemma{lemma_id}")

                # If no sample passed both checks, use first sample (default behavior)
                if len(lemma_selected) == 0:
                    lemma_selected.append({
                        "sample": samples[0],
                        "compilation": compilations[0] if compilations else {
                            "pass": False,
                            "compilation_result": {},
                            "compiled_code": ""
                        },
                        "validation": validations.get(0) if self.validation_enabled and self.validation_type == 'binary' and 0 in validations else None,
                        "sample_idx": 0,
                        "lemma_id": lemma_id,
                        "item_idx": item_idx
                    })
                    failed_count += 1
                    if verbosity >= 1:
                        if self.validation_enabled:
                            if self.validation_type == 'select':
                                logger.warning(f"No LLM-selected sample found for item {item_idx}, lemma{lemma_id}, using first sample")
                            else:
                                logger.warning(f"No compiling+validated sample found for item {item_idx}, lemma{lemma_id}, using first sample")
                        else:
                            logger.warning(f"No compiling sample found for item {item_idx}, lemma{lemma_id}, using first sample")
                else:
                    selected_count += 1

                # Add this lemma's selections to the item's list
                selected_by_item[item_idx].extend(lemma_selected)

            # Now write selected_formalizations for each item
            for item_idx, item_selections in selected_by_item.items():
                item = data_list[item_idx]
                parsed_breakdown = item.get("parsed_breakdown", {})
                parsed_breakdown['selected_formalizations'] = []

                for sel in item_selections:
                    # Determine validation_pass based on validation type
                    if self.validation_enabled:
                        if self.validation_type == 'select':
                            # In select mode, the selected sample is considered valid
                            validation_pass = True
                        else:
                            # In binary mode, check the verdict
                            validation_pass = (sel["validation"].get("verdict") == "yes") if sel["validation"] else False
                    else:
                        validation_pass = True  # No validation means pass by default

                    lemma = {
                        'formalization_id': sel["sample_idx"],
                        'formal_statement': sel["sample"]["formal_statement"],
                        'formal_statement_raw': sel["sample"]["formal_statement_raw"],
                        'prompt': sel["sample"].get("prompt", ""),
                        'compilation_result': sel["compilation"]["compilation_result"],
                        'compiled_code': sel["compilation"]["compiled_code"],
                        'compilation_pass': sel["compilation"]["pass"],
                        'validation_pass': validation_pass,
                        'validation_result': sel["validation"],
                        'id': sel["lemma_id"],
                        'detailed_cost': sel["sample"].get("detailed_cost")  # Add detailed_cost from sample
                    }

                    # Find lemma metadata from parsed_breakdown
                    for parsed_lemma in parsed_breakdown.get("lemmas", []):
                        if parsed_lemma and parsed_lemma.get("id") == sel["lemma_id"]:
                            lemma['statement'] = parsed_lemma.get("statement", "")
                            lemma['assumptions'] = parsed_lemma.get("assumptions", "")
                            lemma['proof'] = parsed_lemma.get("proof", "")
                            lemma['dependencies'] = parsed_lemma.get("dependencies", [])
                            break

                    parsed_breakdown['selected_formalizations'].append(lemma)

            # Calculate total selected formalizations
            total_selected = sum(len(data_list[item_idx].get("parsed_breakdown", {}).get("selected_formalizations", []))
                                for item_idx in selected_by_item.keys())
            logger.info(f"Sample selection: {selected_count} lemmas found compiling samples, {failed_count} used fallback, {total_selected} total formalizations kept")

        else:
            # === USE FORMAL_STATEMENT FROM PARSED BREAKDOWN ===
            # Extract formal statements from parsed_breakdown and compile them
            from collections import defaultdict

            logger.info(f"Extracting formal statements from parsed breakdown...")

            compile_tasks = []
            name_to_metadata = {}

            for item_idx, item in enumerate(data_list):
                parsed_breakdown = item.get("parsed_breakdown", {})
                problem_id = item.get("problem_id", "unknown")
                parent_metadata = item.get("metadata", {})

                if not parsed_breakdown or "lemmas" not in parsed_breakdown:
                    if verbosity >= 1:
                        logger.warning(f"No valid parsed_breakdown found for item: {problem_id}")
                    continue

                lemmas = parsed_breakdown.get("lemmas", [])
                for lemma in lemmas:
                    lemma_id = lemma.get("id")
                    # Extract formal_statement from the parsed breakdown
                    formal_statement = lemma.get("formal_statement", "")

                    if not formal_statement or formal_statement == "None":
                        if verbosity >= 1:
                            logger.warning(f"No formal_statement found for item {problem_id}, lemma{lemma_id}")
                        continue

                    # Try extracting from code blocks first
                    extracted_code = extract_code(formal_statement)

                    # If extract_code returns "None", the formal_statement is already plain code
                    # In this case, we need to add the import headers manually
                    if extracted_code == "None":
                        import_head = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
                        extracted_code = import_head + formal_statement

                    # Enforce correct lemma naming
                    formal_statement_corrected = enforce_lemma_name(extracted_code, lemma_id, parent_metadata)

                    if verbosity >= 3 and formal_statement_corrected != extracted_code:
                        logger.debug(f"Corrected lemma name for item {item_idx}, lemma{lemma_id}")

                    if formal_statement_corrected and formal_statement_corrected != "None":
                        code = handle(formal_statement_corrected).strip()

                        # Create metadata for this formalization (formalization_id=0 for non-sampled)
                        lemma_metadata = add_lemma(parent_metadata, lemma_id)
                        lemma_metadata = add_formalization(lemma_metadata, 0)
                        name = generate_uid(lemma_metadata)

                        if verbosity >= 3:
                            logger.debug(f"Preparing to compile {name}, code length: {len(code)}")

                        compile_tasks.append({
                            "name": name,
                            "code": code,
                        })
                        name_to_metadata[name] = {
                            "problem_id": problem_id,
                            "lemma_id": lemma_id,
                            "item_idx": item_idx,
                            "type": "lemma",
                            "metadata": lemma_metadata
                        }

            # Compile all formal statements
            if compile_tasks:
                logger.info(f"Compiling {len(compile_tasks)} formal statements from parsed breakdown...")
                sample_compilation_results = scheduler(compile_tasks, num_workers=self.config.get("cpu", 4))
            else:
                logger.warning("No formal statements to compile")
                sample_compilation_results = []

            # Validate compiled formalizations (if validation enabled)
            validation_map = {}  # {name: validation_result}
            if self.validation_enabled:
                # Prepare validation items for all compiled lemmas
                validation_items = []
                for comp_result in sample_compilation_results:
                    name = comp_result.get("name", "")
                    metadata = name_to_metadata.get(name, {})
                    item_idx = metadata.get("item_idx")
                    lemma_id = metadata.get("lemma_id")
                    is_pass = comp_result.get("compilation_result", {}).get("pass", False)

                    # Only validate samples that compiled successfully
                    if is_pass:
                        item = data_list[item_idx]
                        parsed_breakdown = item.get("parsed_breakdown", {})
                        lemmas = parsed_breakdown.get("selected_formalizations", [])

                        # Find the corresponding lemma data
                        lemma_data = None
                        for lemma in lemmas:
                            if lemma and lemma.get("id") == lemma_id:
                                lemma_data = lemma
                                break

                        if lemma_data:
                            # Build full metadata for this formalization
                            parent_metadata = item.get("metadata", {})
                            lemma_metadata = add_lemma(parent_metadata, lemma_id)
                            lemma_metadata = add_formalization(lemma_metadata, 0)  # formalization_id=0 for non-sampled mode

                            validation_items.append({
                                'problem_id': item.get("problem_id", "unknown"),
                                'lemma_id': lemma_id,
                                'sample_idx': None,  # No sample idx for non-sampled mode
                                'statement': lemma_data.get("statement", ""),
                                'assumption': lemma_data.get("assumptions", ""),
                                'proof': lemma_data.get("proof", ""),
                                'formal_statement': lemma_data.get("formal_statement", ""),
                                'name': name,
                                'original_statement': item.get("informal_prefix", ""),
                                'metadata': lemma_metadata
                            })

                # Run validation
                validation_results = self._validate_formalizations(validation_items)
                all_validation_results.extend(validation_results)

                # Index validation results by name
                for v_result in validation_results:
                    name = v_result.get('name')
                    if name:
                        validation_map[name] = v_result

            # Update lemmas with compilation results
            compilation_map = {}
            for comp_result in sample_compilation_results:
                name = comp_result.get("name", "")
                compilation_map[name] = comp_result

                if verbosity >= 3:
                    logger.debug(f"Compilation result for {name}: pass={comp_result.get('compilation_result', {}).get('pass', False)}, code_len={len(comp_result.get('code', ''))}")

            compiled_count = 0
            failed_count = 0

            for item_idx, item in enumerate(data_list):
                parsed_breakdown = item.get("parsed_breakdown", {})
                problem_id = item.get("problem_id", "unknown")
                parent_metadata = item.get("metadata", {})
                lemmas = parsed_breakdown.get("selected_formalizations", [])

                for lemma in lemmas:
                    lemma_id = lemma.get("id")
                    formalization_id = lemma.get("formalization_id", 0)
                    # Generate name using metadata (same as compilation task)
                    lemma_metadata = add_lemma(parent_metadata, lemma_id)
                    lemma_metadata = add_formalization(lemma_metadata, formalization_id)
                    name = generate_uid(lemma_metadata)

                    if name in compilation_map:
                        comp_result = compilation_map[name]
                        is_pass = comp_result.get("compilation_result", {}).get("pass", False)

                        # Store the formal_statement_raw (original from parsed breakdown)
                        lemma["formal_statement_raw"] = lemma.get("formal_statement", "")
                        lemma["compilation_result"] = comp_result.get("compilation_result", {})
                        lemma["compiled_code"] = comp_result.get("code", "")
                        lemma["compilation_pass"] = is_pass

                        # Mark as formalization 0 (not sampled, using parsed breakdown formal_statement)
                        lemma["formalization_id"] = 0

                        # Add validation results (if validation enabled)
                        if name in validation_map:
                            validation = validation_map[name]
                            lemma["validation_pass"] = (validation.get("verdict") == "yes")
                            lemma["validation_result"] = validation
                        elif self.validation_enabled:
                            # Validation enabled but no result (likely didn't compile)
                            lemma["validation_pass"] = False
                            lemma["validation_result"] = None
                        else:
                            # Validation not enabled, mark as pass by default
                            lemma["validation_pass"] = True
                            lemma["validation_result"] = None

                        if verbosity >= 3:
                            logger.debug(f"Updated lemma {name}: compiled_code_len={len(lemma.get('compiled_code', ''))}")

                        if is_pass:
                            compiled_count += 1
                        else:
                            failed_count += 1
                            if verbosity >= 1:
                                logger.warning(f"Compilation failed for item {problem_id}, lemma{lemma_id}")
                    else:
                        # No compilation result (e.g., no formal_statement was provided)
                        lemma["formal_statement_raw"] = lemma.get("formal_statement", "")
                        lemma["compilation_result"] = {}
                        lemma["compiled_code"] = ""
                        lemma["compilation_pass"] = False
                        lemma["validation_pass"] = False if self.validation_enabled else True
                        lemma["validation_result"] = None
                        failed_count += 1

            logger.info(f"Compilation results: {compiled_count} passed, {failed_count} failed")

            # No cost for this path (not using LLM)
            total_cost = 0

            # STEP: If we have breakdown_parser samples, select the best one per origin problem
            if has_breakdown_parser_samples:
                from collections import defaultdict

                # Group items by origin_problem_id
                samples_by_origin = defaultdict(list)
                for item in data_list:
                    origin_id = item.get("origin_problem_id")
                    if origin_id:
                        samples_by_origin[origin_id].append(item)

                # Select best sample for each origin
                selected_items = []
                for origin_id, samples in samples_by_origin.items():
                    # Find first sample where ALL lemmas compiled
                    best_sample = None
                    for sample in samples:
                        parsed_breakdown = sample.get("parsed_breakdown", {})
                        lemmas = parsed_breakdown.get("lemmas", [])

                        if not lemmas:
                            # No lemmas, can keep this sample
                            best_sample = sample
                            break

                        # Check if all lemmas compiled and validated
                        all_compiled = all(lemma.get("compilation_pass", False) for lemma in lemmas)
                        all_validated = all(lemma.get("validation_pass", True) for lemma in lemmas)

                        if all_compiled and all_validated:
                            best_sample = sample
                            if verbosity >= 2:
                                if self.validation_enabled:
                                    logger.info(f"Selected sample {sample.get('sample_idx')} for {origin_id} (all {len(lemmas)} lemmas compiled+validated)")
                                else:
                                    logger.info(f"Selected sample {sample.get('sample_idx')} for {origin_id} (all {len(lemmas)} lemmas compiled)")
                            break

                    if best_sample:
                        selected_items.append(best_sample)
                    else:
                        if verbosity >= 1:
                            logger.warning(f"No fully-compiled sample found for {origin_id} ({len(samples)} samples tried)")

                # Replace data_list with selected items
                logger.info(f"Sample selection: kept {len(selected_items)}/{len(set(samples_by_origin.keys()))} unique problems")
                data_list = selected_items

        # STEP 6: Add theorem formal statements from dataset (no need to compile, they're from dataset)
        for item in data_list:
            parsed_breakdown = item.get("parsed_breakdown", {})
            theorem = parsed_breakdown.get("theorem", {})

            if theorem:
                dataset_formal_statement = item.get("lean4_code", "")
                if dataset_formal_statement:
                    theorem["formal_statement"] = dataset_formal_statement
                    theorem["formal_statement_raw"] = dataset_formal_statement
                    # Mark theorem as passed (from dataset, assumed correct)
                    theorem["compilation_pass"] = True
                    theorem["validation_pass"] = True  # Assumed correct from dataset
                    if verbosity >= 2:
                        logger.info(f"Using formal_statement from dataset for theorem in {item.get('problem_id', 'unknown')}")
                else:
                    if verbosity >= 1:
                        logger.warning(f"No formal_statement found in dataset for item: {item.get('problem_id', 'unknown')}")

            item["parsed_breakdown"] = parsed_breakdown
            if "formalization_cost" not in item:
                item["formalization_cost"] = 0

        # Store total cost in first item
        if data_list:
            data_list[0]["total_formalization_cost"] = total_cost

        logger.info(f"Formalization queries completed. Total cost: {total_cost}")

        # STEP 7: Filter out items where not all lemmas compiled and validated
        filtered_data_list = []
        for item in data_list:
            parsed_breakdown = item.get("parsed_breakdown", {})
            lemmas = parsed_breakdown.get("selected_formalizations", [])

            if not lemmas:
                # No lemmas, keep item
                filtered_data_list.append(item)
                continue

            # Check if all lemmas compiled and validated
            all_lemmas_compiled = all(lemma.get("compilation_pass", False) for lemma in lemmas)
            all_lemmas_validated = all(lemma.get("validation_pass", True) for lemma in lemmas)

            if all_lemmas_compiled and all_lemmas_validated:
                filtered_data_list.append(item)
                if verbosity >= 2:
                    item_meta = item.get("metadata", {})
                    problem_id = item_meta.get("parent_problem_id") or item_meta.get("origin_problem_id", "unknown")
                    if self.validation_enabled:
                        logger.info(f"Keeping {problem_id}: all {len(lemmas)} lemmas compiled+validated")
                    else:
                        logger.info(f"Keeping {problem_id}: all {len(lemmas)} lemmas compiled")
            else:
                compiled_count = sum(1 for lemma in lemmas if lemma.get("compilation_pass", False))
                validated_count = sum(1 for lemma in lemmas if lemma.get("validation_pass", True))
                if verbosity >= 1:
                    item_meta = item.get("metadata", {})
                    problem_id = item_meta.get("parent_problem_id") or item_meta.get("origin_problem_id", "unknown")
                    if self.validation_enabled:
                        logger.warning(f"Filtering out {problem_id}: only {compiled_count}/{len(lemmas)} lemmas compiled, {validated_count}/{len(lemmas)} validated")
                    else:
                        logger.warning(f"Filtering out {problem_id}: only {compiled_count}/{len(lemmas)} lemmas compiled")

        if self.validation_enabled:
            logger.info(f"Filtered breakdowns: kept {len(filtered_data_list)}/{len(data_list)} (all lemmas compiled+validated)")
        else:
            logger.info(f"Filtered breakdowns: kept {len(filtered_data_list)}/{len(data_list)} (all lemmas compiled)")
        
        # data_list = filtered_data_list

        # Save results
        base_output_dir = self.global_config.get("output_dir")
        output_dir = os.path.join(base_output_dir, f"round{round_num}", "formalizer")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "formalized.json")
        jsave(data_list, output_file)

        # Save compilation results separately (from sample selection)
        compilation_file = os.path.join(output_dir, "compilation_results.json")
        jsave(sample_compilation_results, compilation_file)

        # Save validation results separately (if validation was performed)
        if self.validation_enabled and all_validation_results:
            validation_file = os.path.join(output_dir, "validation_results.json")
            jsave(all_validation_results, validation_file)
            if verbosity >= 1:
                logger.info(f"Saved {len(all_validation_results)} validation results to {validation_file}")

        # Save selection results separately (if select mode was used)
        if self.validation_enabled and self.validation_type == 'select' and all_selection_results:
            selection_file = os.path.join(output_dir, "selection_results.json")
            jsave(all_selection_results, selection_file)
            if verbosity >= 1:
                logger.info(f"Saved {len(all_selection_results)} selection results to {selection_file}")

        # Save lemmas and theorems as separate problems
        self._save_lemma_problems(filtered_data_list, round_num)

        # Cleanup VLLM model to free GPU memory
        self._cleanup_model()

        logger.info(f"Formalization complete. Results saved to {output_file}")

        return data_list
