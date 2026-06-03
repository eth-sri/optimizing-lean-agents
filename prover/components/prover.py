import os
import json
import random
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from prover.core.component import Component
from lean_compiler.repl_scheduler import scheduler
from prover.utils import DeepSeekCoTHandler, DeepSeekNonCoTHandler, KiminaCoTHandler, get_error_str
from prover.utils import process_for_correction
from prover.utils import problem_check, extract_code
from prover.query.api import APIQuery
from jload import jload, jsave
from metadata_utils import add_attempt, add_correction, generate_uid, copy_metadata
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


class ProverComponent(Component):
    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)

        self.base_template_path = f"prover/template/prover/{self.config.get('template')}.md"
        self.base_axiom_template_path = f"prover/template/prover/{self.config.get('template')}_axiom.md"

        self.use_informal_summ = self.config.get('use_informal_summ', False)
        if self.use_informal_summ:
            self.correction_template_path = f"prover/template/prover/{self.config.get('template')}_correction_informal_summary.md"
        else:
            self.correction_template_path = f"prover/template/prover/{self.config.get('template')}_correction.md"

        self.base_template = self._load_template(self.base_template_path)
        # self.base_axiom_template = self._load_template(self.base_axiom_template_path)
        self.correction_template = self._load_template(self.correction_template_path)

        # use_correction can be:
        # - False: never use correction template
        # - int (0, 1, 2, ...): start using correction template from that round onwards
        self.use_correction = self.config.get('use_correction', True)

        # use_metadata controls whether to use metadata/uid based IDs (new) or problem_id (legacy)
        # Default: False for backwards compatibility
        self.use_metadata = self.config.get('use_metadata', False)

    def _should_use_correction(self, round_num):
        """
        Determine whether to use correction template based on round_num and use_correction config.

        Args:
            round_num: Current correction round number

        Returns:
            bool: True if correction template should be used
        """
        if self.use_correction is False:
            # Never use correction
            return False
        elif isinstance(self.use_correction, int):
            # Use correction starting from the specified round
            return round_num >= self.use_correction
        else:
            # Default behavior (True or other truthy value): use correction for round_num > 0
            return round_num > 0

    def process(self, data_list, round_num=0):
        verbosity = self.global_config.get('verbosity', 3)

        if verbosity >= 1:
            print(f"ProverComponent: Processing {len(data_list)} problems, round {round_num}")

        data_list = self._preprocess(data_list, round_num)

        data_proofs = self._generate_proof(data_list, round_num)

        data_compilation = self._compile(data_proofs, round_num)

        data_output = self._post_process(data_proofs, data_compilation, round_num + 1)

        return data_output

    def _preprocess(self, data_list, round_num):
        data_for_processing = []
        if round_num == 0:
            sample_times = self.global_config.get('pipeline').get('initial_samples')
        else:
            sample_times = self.global_config.get('pipeline').get('correction_samples')

        for item in data_list:
            # Check if item has new metadata format
            metadata = item.get("metadata")

            if metadata:
                # NEW FORMAT: Use structured metadata
                if round_num == 0:
                    # Round 0: Create multiple samples per problem
                    for n in range(sample_times):
                        item_copy = item.copy()

                        # Add attempt_id to metadata
                        new_metadata = add_attempt(metadata, n)
                        # In round 0, initial_attempt_id equals attempt_id
                        new_metadata["initial_attempt_id"] = n

                        item_copy["metadata"] = new_metadata
                        item_copy["uid"] = generate_uid(new_metadata)

                        data_for_processing.append(item_copy)
                else:
                    # Correction rounds: Create correction_samples attempts per failed problem
                    for n in range(sample_times):
                        item_copy = item.copy()

                        # Update attempt_id and correction_round_id in metadata
                        new_metadata = copy_metadata(metadata)
                        new_metadata["attempt_id"] = n
                        new_metadata["correction_round_id"] = round_num
                        # Preserve initial_attempt_id from previous round (with fallback for old data)
                        new_metadata["initial_attempt_id"] = metadata.get("initial_attempt_id", metadata.get("attempt_id", 0))

                        item_copy["metadata"] = new_metadata
                        item_copy["uid"] = generate_uid(new_metadata)

                        data_for_processing.append(item_copy)
            else:
                # OLD FORMAT: Fallback to legacy problem_id + id_maps approach
                original_id = item.get("origin_problem_id", item.get("problem_id", item.get("name")))
                problem_id = item.get("problem_id")

                for n in range(sample_times):
                    item_copy = item.copy()
                    if problem_id.rsplit("_", 1)[0].endswith("_th"):
                        type = "theorem"
                    else:
                        type = "lemma"
                    id_maps = item_copy.get("id_maps", [])
                    item_copy["origin_problem_id"] = original_id
                    item_copy["problem_id"] = f"{problem_id}_p{n}"
                    item_copy["type"] = type

                    if round_num > 0:
                        item_copy["id_maps"] = id_maps + [
                            {f"corr{round_num}_proof_id": item_copy["problem_id"]}
                        ]
                    else:
                        item_copy["id_maps"] = id_maps + [
                            {"proof_id": item_copy["problem_id"]}
                        ]

                    data_for_processing.append(item_copy)

        return data_for_processing

    def _generate_proof(self, data_list, round_num):
        verbosity = self.global_config.get('verbosity', 3)

        if verbosity >= 1:
            print(f"Generating proofs")
        output_dir = self.global_config.get('output_dir')

        model = self.model_config.get('model')
        api = self.model_config.get('api')
        max_tokens = self.model_config.get('max_tokens', 40960)
        temperature = self.model_config.get('temperature', 1.0)

        # Get model config path for tracking which model was used
        model_config_path = self.config.get('model_config', None)

        kwargs = self.model_config.copy()
        del kwargs["model"]
        del kwargs["api"]
        if 'date' in kwargs:
            del kwargs['date']
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = temperature

        # Get verbosity from global config
        verbosity = self.global_config.get('verbosity', 3)

        querier = APIQuery(
            model=model,
            api=api,
            verbosity=verbosity,
            **kwargs
        )
        
        error_thres = self.config.get('error_thres', True)
        history_mode = self.config.get('history_mode')
        
        # Load tokenizer
        if api == "vllm":
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        else:
            tokenizer = None
        
        # Setup output files
        records_file_suffix = f"_corr{round_num}" if round_num > 0 else ""
        output_file_path_records = os.path.join(output_dir, f'full_records{records_file_suffix}.json')
        output_file_path_inference_codes = os.path.join(output_dir, f'to_inference_codes{records_file_suffix}.json')
        
        # Handle correction rounds - load failed problems
        items_for_llm_processing = data_list
        if verbosity >= 2:
            print(f"Total items for LLM: {len(items_for_llm_processing)} for round {round_num}.")


        all_processed_records = []
        all_inference_code_outputs = []

        records = []
        # Use tqdm only if verbosity >= 1
        iterator = items_for_llm_processing
        if verbosity >= 1:
            iterator = tqdm(items_for_llm_processing, desc=f"Preparing data...")

        for i, item_data in enumerate(iterator):
            # Prepare problem statement
            item_data["lean4_code"] = item_data["lean4_code"].split(":= by")[0] + ":= by sorry"
            

            if not self._should_use_correction(round_num):
            #     if item_data["type"] == "theorem":
            #         template = self.base_axiom_template
            #     else:
                template = self.base_template
            else:
                template = self.correction_template
            prompt_str, messages_for_this = self._prepare_prompt(round_num, item_data, template, tokenizer)

            if api == "vllm":
                num_tokens = len(tokenizer.tokenize(prompt_str))
            else:
                num_tokens = 0
            # Extract the actual prompt text from messages (last user message)
            prompt_text = ""
            for msg in reversed(messages_for_this):
                if msg.get("role") == "user":
                    prompt_text = msg.get("content", "")
                    break

            records.append({
                "token_nums": num_tokens,
                "prompts_for_vllm": messages_for_this,
                "messages_lists_for_current_prompts": messages_for_this,
                "prompt_text": prompt_text,
                "input_items": item_data
            })

        # Filter by token length
        df_med = pd.DataFrame(records)
        max_length = max_tokens * 3 / 4  # fixed to be Qwen
        to_process_df = df_med[df_med.token_nums <= max_length].reset_index(drop=True)
        if verbosity >= 2:
            print(f"In total {len(df_med)}, selected {len(to_process_df)} whose length is smaller than {max_length}")

        for i, response, detailed_cost in querier.run_queries(to_process_df.prompts_for_vllm):
            if verbosity >= 3:
                print(detailed_cost)
            input_item = to_process_df.input_items[i].copy()
            input_item["model_input"] = to_process_df.prompts_for_vllm[i]
            input_item["messages_history_for_this_attempt"] = to_process_df.messages_lists_for_current_prompts[i]
            
            input_item["model_output"] = response
            input_item["model_config_path"] = model_config_path
            extracted_code = extract_code(response)
            if extracted_code == "None" or extracted_code is None:
                input_item["full_code"] = "None"
            else:
                input_item["full_code"] = problem_check(input_item["lean4_code"], extracted_code)

            all_processed_records.append(input_item)

            # Build output record conditionally based on use_metadata flag
            if self.use_metadata:
                # NEW: Use metadata/uid based approach (clean, no redundant fields)
                output_record = {
                    "uid": input_item.get("uid"),
                    "metadata": input_item.get("metadata"),
                    "lean4_code": input_item["lean4_code"],
                    "prompt": to_process_df.prompt_text[i],
                    "model_input": input_item["model_input"],
                    "messages_history_list": input_item["messages_history_for_this_attempt"],
                    "model_output": input_item["model_output"],
                    "full_code": input_item["full_code"],
                    "detailed_cost": detailed_cost,
                    "model_config_path": model_config_path,
                }
            else:
                # LEGACY: Use old problem_id based approach
                output_record = {
                    "problem_id": input_item["problem_id"],
                    "origin_problem_id": input_item.get("origin_problem_id"),
                    "id_maps": input_item.get("id_maps"),
                    "lean4_code": input_item["lean4_code"],
                    "prompt": to_process_df.prompt_text[i],
                    "model_input": input_item["model_input"],
                    "messages_history_list": input_item["messages_history_for_this_attempt"],
                    "model_output": input_item["model_output"],
                    "full_code": input_item["full_code"],
                    "detailed_cost": detailed_cost,
                    "model_config_path": model_config_path,
                }

            all_inference_code_outputs.append(output_record)

        # Clean up records if using metadata (remove redundant legacy fields)
        if self.use_metadata:
            cleaned_records = []
            for record in all_processed_records:
                # Remove redundant fields that are already in metadata
                cleaned = {k: v for k, v in record.items()
                          if k not in ['origin_problem_id', 'problem_id', 'id_maps', 'type']}
                cleaned_records.append(cleaned)
            all_processed_records = cleaned_records

        if verbosity >= 2:
            print(f"Saving round {round_num} to {output_dir}")
        jsave(all_processed_records, output_file_path_records)
        jsave(all_inference_code_outputs, output_file_path_inference_codes)

        if verbosity >= 2:
            print(f"Outputs saved: \n  Records: {output_file_path_records}\n  Inference Codes: {output_file_path_inference_codes}")

        return all_inference_code_outputs

    def _prepare_prompt(self, round_num, data, template, tokenizer):
        if not self._should_use_correction(round_num):
            lean4_code = data["lean4_code"]
            formal_statement = lean4_code.split(":= by")[0] + ":= by sorry"
            informal_prefix = data.get("informal_prefix", "")
            prompt = template.format(formal_statement=formal_statement,
                                     informal_prefix=informal_prefix)

            messages = [{"role": "user", "content": prompt}]
        else:
            error_thres = self.config.get('error_thres', True)

            # Extract errors, handling various data formats
            errors_data = data.get('errors_for_compiled_code_from_prev_round', {})
            if isinstance(errors_data, dict):
                errors = errors_data.get('errors', [])
            elif isinstance(errors_data, list):
                errors = errors_data
            else:
                # If it's a string or other type, wrap it in a list
                errors = [str(errors_data)] if errors_data else []

            error_str = get_error_str(
                    data.get('compiled_code_that_failed_in_prev_round', ''),
                    errors,
                    error_thres
                )
            history_messages_from_prev_round = data.get("history_messages_from_prev_round_for_new_prompt", [])
            prev_round_llm_raw_output = data.get("prev_round_llm_raw_output_for_new_prompt", "")
            error_message_for_prev_round = error_str
            prev_round_code = data.get("compiled_code_that_failed_in_prev_round")


            history_mode = self.config.get('history_mode')
            if history_mode == "all":
                messages = list(history_messages_from_prev_round)
            elif history_mode == "last_one":
                messages = [history_messages_from_prev_round[-1]]

            # if include the COTs from previous rounds
            with_cot = self.config.get('with_cot', True)
            if with_cot:
                assistant_content = prev_round_llm_raw_output
            else:
                assistant_content = prev_round_code
            messages.append({"role": "assistant", "content": assistant_content})

            informal_prefix = data.get("informal_prefix", "")
            if self.config.get('use_informal_summ', False):
                summary = data.get("informal_summary")
                prompt = template.format(prev_round_num=round_num-1,
                                         error_message_for_prev_round=error_message_for_prev_round,
                                         summary=summary,
                                         informal_prefix=informal_prefix)
            else:
                prompt = template.format(prev_round_num=round_num-1,
                                         error_message_for_prev_round=error_message_for_prev_round,
                                         informal_prefix=informal_prefix)
            messages.append({"role": "user", "content": prompt})

        if tokenizer:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = ""

        return text, messages

    def _compile(self, data_list, round_num=0):
        verbosity = self.global_config.get('verbosity', 3)

        if verbosity >= 1:
            print(f"CompilationComponent: Compiling {len(data_list)} problems, round {round_num}")
        if verbosity >= 2:
            print(f"  CPU cores: {self.config.get('cpu', 32)}")

        # Get configuration
        output_dir = self.global_config.get('output_dir')

        cpu_cores = self.config.get('cpu', 32)
        
        if not data_list:
            print("No data to compile")
            return data_list
        
        codes_for_compilation = []
        failed_compilation_results = []

        for item in data_list:
            # Handle items without generated code (e.g., model hit token limit)
            if not item.get("full_code") or item.get("full_code") == "None":
                # Create synthetic failed compilation result so item appears in comp_lookup
                # This allows it to be picked up in correction rounds
                identifier = item.get("uid", item.get("problem_id", item.get("name", "unknown")))
                failed_result = {
                    "name": identifier,
                    "code": "",  # No code was generated
                    "compilation_result": {
                        "pass": False,
                        "complete": False,
                        "errors": ["Model output was None or empty (likely token limit exceeded)"],
                        "warnings": []
                    }
                }
                if self.use_metadata:
                    failed_result["uid"] = item.get("uid")
                    failed_result["metadata"] = item.get("metadata")
                if not self.use_metadata:
                    failed_result["problem_id"] = item.get("problem_id", item.get("name", "unknown"))

                failed_compilation_results.append(failed_result)
                if verbosity >= 2:
                    print(f"Item {identifier}: Model output was None, marking as failed compilation")
                continue

            # Prepare compilation entry conditionally based on use_metadata flag
            if self.use_metadata:
                # NEW: Use metadata/uid based approach (clean, no redundant fields)
                identifier = item.get("uid", "unknown")
                compilation_entry = {
                    "name": identifier,  # Required by scheduler
                    "code": handle(item["full_code"]),
                    "uid": item.get("uid"),
                    "metadata": item.get("metadata")
                }
            else:
                # LEGACY: Use old problem_id based approach
                compilation_entry = {
                    "name": item.get("problem_id", item.get("name", "unknown")),
                    "code": handle(item["full_code"]),
                    "problem_id": item.get("problem_id", item.get("name", "unknown"))
                }

            codes_for_compilation.append(compilation_entry)
        
        if not codes_for_compilation:
            if verbosity >= 1:
                print("No valid code found for compilation")
            # Save any failed compilation results (items with no code generated)
            file_suffix = f"_corr{round_num}" if round_num > 0 else ""
            output_file_path = os.path.join(output_dir, f'code_compilation_repl{file_suffix}.json')
            jsave(failed_compilation_results, output_file_path)
            if failed_compilation_results and verbosity >= 2:
                print(f"Saved {len(failed_compilation_results)} synthetic failed compilation results")
            return data_list

        if verbosity >= 2:
            print(f"Preparing {len(codes_for_compilation)} items for compilation")

        # Shuffle codes as in original compile.py
        random.shuffle(codes_for_compilation)

        try:
            # Run compilation using the scheduler
            compilation_results = scheduler(codes_for_compilation, num_workers=cpu_cores)
            if verbosity >= 2:
                print(f"Compilation completed. Got {len(compilation_results)} results")

            # Merge synthetic failed results with actual compilation results
            all_compilation_results = failed_compilation_results + compilation_results

            # Save compilation results
            file_suffix = f"_corr{round_num}" if round_num > 0 else ""
            output_file_path = os.path.join(output_dir, f'code_compilation_repl{file_suffix}.json')
            jsave(all_compilation_results, output_file_path)
            if verbosity >= 2:
                print(f"Compilation results saved to: {output_file_path}")
                if failed_compilation_results:
                    print(f"  Included {len(failed_compilation_results)} synthetic failed results")

            return all_compilation_results

        except Exception as e:
            print(f"Error during compilation: {e}")
            import traceback
            traceback.print_exc()

            # Save failed compilation results even on error
            file_suffix = f"_corr{round_num}" if round_num > 0 else ""
            output_file_path = os.path.join(output_dir, f'code_compilation_repl{file_suffix}.json')
            jsave(failed_compilation_results, output_file_path)

            return []

    def _post_process(self, data_proofs, data_compilation, round_num=0):
        base_output_template = (
            self.model_config
                .get('base_output_template', '')
        )
        print("base_output_template:", base_output_template)

        # Get correct_all_failed_attempts from pipeline config, default to False
        correct_all_failed_attempts = self.global_config.get('pipeline', {}).get('correct_all_failed_attempts', False)

        processed_data = process_for_correction(
            data_proofs, data_compilation,
            round_num, base_output_template,
            base_output_dir=None,
            use_metadata=self.use_metadata,
            correct_all_failed_attempts=correct_all_failed_attempts
        )

        return processed_data

    def _load_template(self, template_path):

        if not template_path or not os.path.exists(template_path):
            raise ValueError(f"Template path {template_path} does not exist or is not set in config")

        with open(template_path, "r") as f:
            template = f.read()

        return template