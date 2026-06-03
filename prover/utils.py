# adapted from https://github.com/Goedel-LM/Goedel-Prover-V2.git

import re
import pandas as pd
import numpy as np
import json
from jload import jload, jsave
import os
import re
from id_utils import get_origin_problem_id


def problem_check(statement, full_code):
    full_code = replace_statement_in_proof(statement, full_code)        
    return full_code

def extract_code(inputs):
    import_head = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"       
    pattern = r'```lean4\n(.*?)\n```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r'```lean4\n(.*?)```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    pattern = r'```lean\n(.*?)```'
    matches = re.findall(pattern, inputs, re.DOTALL)
    if matches:
        return import_head + matches[-1]
    return "None"

def get_error_str(code, errors, error_thres):
    err_str = ""
    code_lines = code.split('\n')
    token_lengths = [len(line) + 1 for line in code_lines]

    # error_thres = False

    # error_num_thres = 8 if error_thres else error_num_thres
    error_num_thres = 8 if error_thres else len(errors)

    for i, error in enumerate(errors[:error_num_thres]):
        # Handle case where error is a string instead of a dictionary
        if isinstance(error, str):
            err_str += f"\nError {i + 1}: {error}\n"
            continue

        # Handle case where error is malformed (missing required keys)
        if not isinstance(error, dict) or 'pos' not in error:
            err_str += f"\nError {i + 1}: {str(error)}\n"
            continue

        start_line = error['pos']['line'] - 1
        start_col = error['pos']['column']

        if error['endPos'] is None:
            end_line = start_line
            end_col = len(code_lines[start_line])
        else:
            end_line = error['endPos']['line'] - 1
            end_col = error['endPos']['column']

        start_char_pos = sum(token_lengths[:start_line]) + start_col
        end_char_pos = sum(token_lengths[:end_line]) + end_col
        
        err_str += f"\nError {i + 1}:\n"
        err_str += f"\nCorresponding Code:\n```lean4\n"
        
        error_code = ""
        for ii in range(-4, 0):
            if start_line + ii >= 0:
                error_code += f"{code_lines[start_line + ii]}\n"
        if start_line != end_line:
            error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:] + "\n"
            
            if not error_thres:
                for j in range(start_line + 1, end_line):
                    error_code += f"{code_lines[j]}\n"
            else:
                show_line = 6
                for j in range(start_line + 1, min(end_line, start_line + show_line)):
                    error_code += f"{code_lines[j]}\n"
                if end_line > start_line + show_line:
                    leading_spaces = len(code_lines[j]) - len(code_lines[j].lstrip(' '))
                    error_code += "\n" + " " * leading_spaces + "... --[Truncated]-- ...\n"

            error_code += code_lines[end_line][:end_col] + "</error>" + code_lines[end_line][end_col:] + "\n"
        else:
            error_code += code_lines[start_line][:start_col] + "<error>" + code_lines[start_line][start_col:end_col] + "</error>" + code_lines[start_line][end_col:] + "\n"
        if end_line + 1 < len(code_lines):
            error_code += f"{code_lines[end_line + 1]}\n"
            
        err_str += error_code
        err_str += f"\n```\n"
        err_str += f"\nError Message: {error['data']}\n"
    
    if len(errors) > error_num_thres:
        err_str += f"\n... [Omitted {len(errors) - error_num_thres} more errors] ...\n"
        
    return err_str

def extract_dpsk_instruction(dpsk_str): # dpsk 7b output
    return  dpsk_str.split("<｜User｜>")[1].split("<｜Assistant｜>")[0]

def extract_qwen_instruction(qwen_str): # qwen output
    return  qwen_str.split("<|im_start|>user")[1].split("<|im_end|>")[0].strip()

def load_data_for_correction(base_output_dir_for_prev_round: str, current_correction_round_num: int,
        base_output_template: str, use_metadata: bool = False, correct_all_failed_attempts: bool = False):
    print(
        f"Loading data for correction round {current_correction_round_num} from base directory: {base_output_dir_for_prev_round}")

    if current_correction_round_num == 1:
        prev_round_suffix = ""  # R0 files have no suffix
    elif current_correction_round_num > 1:
        prev_round_suffix = f"_corr{current_correction_round_num - 1}"
    else:
        print("Error: load_data_for_correction called with invalid current_correction_round_num (must be >= 1).")
        return []

    prev_inference_file = os.path.join(base_output_dir_for_prev_round, f"to_inference_codes{prev_round_suffix}.json")
    prev_compilation_file = os.path.join(base_output_dir_for_prev_round,
                                         f"code_compilation_repl{prev_round_suffix}.json")

    assert prev_inference_file, f"Error: Required previous inference file not found: {prev_inference_file}"
    assert prev_compilation_file, f"Error: Required previous compilation file not found: {prev_compilation_file}"


    to_inference_data_prev_round = jload(prev_inference_file)
    compilation_results_data_prev_round = jload(prev_compilation_file)

    return process_for_correction(to_inference_data_prev_round, compilation_results_data_prev_round,
                                   current_correction_round_num, base_output_template, base_output_dir_for_prev_round, use_metadata, correct_all_failed_attempts)

def process_for_correction(to_inference_data_prev_round, compilation_results_data_prev_round,
        current_correction_round_num, base_output_template, base_output_dir=None, use_metadata: bool = False, correct_all_failed_attempts: bool = False):

    if base_output_template == "qwen":
        extract_fun = extract_qwen_instruction
    elif base_output_template == "dpsk":
        extract_fun = extract_dpsk_instruction
    elif base_output_template == "api":
        extract_fun = lambda x: x.get("content", "")
    else:
        print("unsupported base template")
        raise Exception

    print(len(to_inference_data_prev_round), "to_inference_data_prev_round items loaded.")

    if "messages_history_list"  not in to_inference_data_prev_round[0]:
        for d in to_inference_data_prev_round:
            # print(d["model_input"])
            d["messages_history_list"] = [{"role": "user", "content": extract_fun(d["model_input"])}]


    comp_lookup = {r["name"]: {"result": r["compilation_result"], "code": r["code"]}
                   for r in compilation_results_data_prev_round if
                   isinstance(r, dict) and "name" in r and "compilation_result" in r and "code" in r}

    failed_problem_variants = {}

    if use_metadata:
        passed_problem_keys = set()
        # NEW FORMAT: Use uid and metadata for grouping
        from metadata_utils import generate_problem_key

        for item_prev_round in to_inference_data_prev_round:
            uid_variant = item_prev_round.get("uid")
            metadata = item_prev_round.get("metadata")

            if not uid_variant or not metadata:
                continue

            # Generate problem key (excludes attempt_id, iteration_id, correction_round_id)
            problem_key = generate_problem_key(metadata)

            if uid_variant in comp_lookup:
                comp_data = comp_lookup[uid_variant]

                is_pass = comp_data["result"].get("pass", False)
                is_complete = comp_data["result"].get("complete", False)

                if is_pass and is_complete:
                    passed_problem_keys.add(problem_key)
                else:
                    if problem_key not in failed_problem_variants:
                        failed_problem_variants[problem_key] = []

                    failed_problem_variants[problem_key].append({
                        "lean4_code": item_prev_round["lean4_code"],
                        "compiled_code_that_failed_in_prev_round": comp_data["code"],
                        "errors_for_compiled_code_from_prev_round": comp_data["result"],
                        "prev_round_llm_raw_output_for_new_prompt": item_prev_round.get("model_output", ""),
                        "history_messages_from_prev_round_for_new_prompt": item_prev_round.get("messages_history_list", []),
                        "metadata": metadata
                    })
    else:
        # LEGACY FORMAT: Use problem_id and origin_problem_id
        passed_original_ids = set()

        for item_prev_round in to_inference_data_prev_round:
            problem_id_variant = item_prev_round.get("problem_id")
            original_problem_id = item_prev_round.get("origin_problem_id")

            if not problem_id_variant or not original_problem_id: continue
            id_maps = item_prev_round.get("id_maps")
            if id_maps is None:
                assert current_correction_round_num == 1, "Only first revision round accepts no id maps input. Please check your input data."
                id_maps = [{"origin_problem_id": original_problem_id}, {"generation_id": problem_id_variant}]

            if problem_id_variant in comp_lookup:
                comp_data = comp_lookup[problem_id_variant]

                is_pass = comp_data["result"].get("pass", False)
                is_complete = comp_data["result"].get("complete", False)

                if is_pass and is_complete:
                    passed_original_ids.add(original_problem_id)
                else:
                    if original_problem_id not in failed_problem_variants:
                        failed_problem_variants[original_problem_id] = []

                    failed_problem_variants[original_problem_id].append({
                        "last_problem_id": problem_id_variant,
                        "origin_problem_id": original_problem_id,
                        "id_maps": id_maps, 
                        "lean4_code": item_prev_round["lean4_code"],
                        "compiled_code_that_failed_in_prev_round": comp_data["code"],
                        "errors_for_compiled_code_from_prev_round": comp_data["result"],
                        "prev_round_llm_raw_output_for_new_prompt": item_prev_round.get("model_output", ""),
                        "history_messages_from_prev_round_for_new_prompt": item_prev_round.get("messages_history_list", [])
                    })

    data_for_new_correction_attempts = []
    total_variants = 0
    unique_p = 0
    for problem_key, variants in failed_problem_variants.items():
        # Check against the appropriate passed set based on mode
        # ONLY skip if correct_all_failed_attempts is False AND the problem has at least one passing attempt
        if not correct_all_failed_attempts:
            if use_metadata:
                if problem_key in passed_problem_keys:
                    continue
            else:
                if problem_key in passed_original_ids:
                    continue

        unique_p += 1
        total_variants += len(variants)
        for variant_item in variants:
            new_attempt_item = variant_item.copy()

            if use_metadata:
                # NEW FORMAT: Add correction_round_id to metadata
                if "metadata" in new_attempt_item and new_attempt_item["metadata"] is not None:
                    from metadata_utils import add_correction, generate_uid
                    new_metadata = add_correction(new_attempt_item["metadata"], current_correction_round_num)
                    new_attempt_item["metadata"] = new_metadata
                    new_attempt_item["uid"] = generate_uid(new_metadata)
                    # Do NOT include problem_id and id_maps when using metadata
                    new_attempt_item.pop("problem_id", None)
                    new_attempt_item.pop("id_maps", None)
                else:
                    print(f"Warning: use_metadata=True but no metadata found for problem, skipping")
                    continue
            else:
                # LEGACY FORMAT: Append correction to problem_id string
                problem_id_variant = variant_item["last_problem_id"]
                new_attempt_item["problem_id"] = f"{problem_id_variant}_corr{current_correction_round_num}"

            data_for_new_correction_attempts.append(new_attempt_item)

    mode_str = "ALL failed attempts" if correct_all_failed_attempts else "problems with NO passing attempts"
    print(f"Correction Round {current_correction_round_num}: Correcting {mode_str}. Identified {unique_p} unique problems with {total_variants} failed variants. " f"Generating {len(data_for_new_correction_attempts)} new samples for LLM inference.")
    return data_for_new_correction_attempts

def remove_comments(text): # remove comments
    # First remove all /- ... -/ blocks
    text = re.sub(r'/-.*?-/', '', text, flags=re.DOTALL)
    # text = re.sub(r'/- (?!special open -/).*?-/', '', text, flags=re.DOTALL)
    # text = re.sub(r'/-{1,2} (?!special open -/).*?-{1,2}/', '', text, flags=re.DOTALL)
    # Then remove -- comments from each line
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # Split on -- and keep only the first part
        cleaned_line = line.split('--', 1)[0]
        cleaned_lines.append(cleaned_line)
    # Join back together and remove excessive empty lines
    cleaned_text = '\n'.join(cleaned_lines)
    # Remove multiple consecutive empty lines
    # cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    return cleaned_text.strip()

def return_theorem_to_prove(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    pattern = r'((?:theorem|lemma).*?:=\s*by\s*sorry)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None


def return_theorem_to_replace(text):
    # Pattern that matches from 'theorem' or 'lemma' to ':= by sorry' with any content in between
    # pattern = r'((?:theorem).*?:=\s*by)'
    pattern = r'((?:^|\s)(?:theorem|lemma)\s+.*?:=\s*by)'
    match = re.search(pattern, text, re.DOTALL)
    return match.span() if match else None

def replace_statement_in_proof(statement, proof):
    if ("apply?" in proof) or ("exact?" in proof):
        return F"**Error**, 'apply?' or 'exact?' is used, which is not allowed."
    stats_re = remove_comments(statement)
    stats_span_= return_theorem_to_prove(stats_re)
    if stats_span_ is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in statement.split('\n')])
        return F"**Error**, can not find 'theorem'/'lemma' and ':= sorry' in {error_app}"
    proof_str = remove_comments(proof)
    span = return_theorem_to_replace(proof_str)
    if span is None:
        error_app = '\n'.join(["\n"] + ['-- ' + x for x in proof.split('\n')])
        return F"**Error**, can not find 'theorem'/'lemma' and ':=' in {error_app}"
    return stats_re[:stats_span_[1]].replace("sorry", "") + proof_str[span[1]:]


class InferenceHandler:
    # Constructor
    def __init__(self):
        pass
    
    def extrac_code(self, inputs):
        pattern = r'```lean4\n(.*?)\n```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return matches[-1]
        pattern = r'```lean4\n(.*?)```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return matches[-1]
        pattern = r'```lean\n(.*?)```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return matches[-1]
        return "None"


    def clean_code_string(self, code_string):
        # Split the code string into lines
        lines = code_string.splitlines()
        
        # Filter out lines that start with specified keywords or are blank
        filtered_lines = [
            line for line in lines 
            if not (line.startswith("import") or line.startswith("set_option") or line.startswith("open") or line.strip() == "")
        ]
        
        # Join the remaining lines back into a single string
        cleaned_code = "\n".join(filtered_lines)
        return cleaned_code

    def prover_inference(self, lean4_code, tokenizer):
        pass  # This method must be implemented by any derived class

    def generate_correction_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,
                                   prev_round_llm_raw_output,
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num):
        # Returns (prompt_str, messages_list_for_this_prompt)
        raise NotImplementedError


    def split_list_into_chunks(self, input_list, num_chunks):
        """Split a list into approximately equal-sized chunks using only Python built-ins."""
        # Make sure input_list is a regular Python list
        input_list = list(input_list)
        
        # Calculate the length of the list
        list_length = len(input_list)
        
        # Calculate the base size for each chunk
        base_chunk_size = list_length // num_chunks
        
        # Calculate how many chunks need an extra element
        # (when the list can't be evenly divided)
        remainder = list_length % num_chunks
        
        chunks = []
        index = 0
        
        # Create each chunk
        for i in range(num_chunks):
            # Determine this chunk's size (add an extra element if needed)
            current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
            
            # If we've reached the end of the list or this chunk would be empty, stop
            if index >= list_length or current_chunk_size == 0:
                break
                
            # Add this chunk to our result
            chunks.append(input_list[index:index + current_chunk_size])
            index += current_chunk_size
        
        return chunks
    
    def load_split(self, input_file, split):
        # data_list = []
        if input_file.endswith('.jsonl'):
            df = pd.read_json(input_file, lines=True)
            if split == "none":
                return df.to_dict(orient='records')
            else:
                return df[df.split.apply(lambda x: str(x) == str(split))].to_dict(orient='records')
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
            if split == "none":
                return df.to_dict(orient='records')
            else:
                return df[df.split.apply(lambda x: str(x) == str(split))].to_dict(orient='records')
    
    def problem_check(self,statement, full_code):
        
        return full_code


class DeepSeekCoTHandler(InferenceHandler):
    def __init__(self):
        pass 

    def extrac_code(self, inputs):
        import_head = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"       
        pattern = r'```lean4\n(.*?)\n```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return import_head + matches[-1]
        pattern = r'```lean4\n(.*?)```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return import_head + matches[-1]
        pattern = r'```lean\n(.*?)```'
        matches = re.findall(pattern, inputs, re.DOTALL)
        if matches:
            return import_head + matches[-1]
        return "None"

    def prover_inference(self, lean4_code, tokenizer):
        formal_statement = lean4_code.split(":= by")[0] + ":= by sorry" # include sorry https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B
        prompt = F"Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}```\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."
        messages = [
            {"role": "user", "content": prompt}
        ]
        if tokenizer:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = ""
        return text, messages
    
    def problem_check(self,statement, full_code):
        full_code = replace_statement_in_proof(statement, full_code)        
        return full_code

    def generate_correction_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,
                                   prev_round_llm_raw_output,
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num,
                                   history_mode="all"):
        original_stmt_for_prompt = lean4_code_original_stmt.split(":= by")[0] + ":= by sorry"

        if history_mode == "all":
            current_messages = list(history_messages_from_prev_round)
        elif history_mode == "last_one":
            current_messages = [history_messages_from_prev_round[-1]]

        # Add PREVIOUS assistant's (failed) attempt
        assistant_content = prev_round_llm_raw_output
        current_messages.append({"role": "assistant", "content": assistant_content})

        # Add CURRENT user feedback and request for new attempt
        user_feedback_content = (
            f"The proof (Round {current_correction_round_num - 1}) is not correct. Following is the compilation error message, where we use <error></error> to signal the position of the error.\n\n{error_message_for_prev_round}"
            "\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error message."
        )
        current_messages.append({"role": "user", "content": user_feedback_content})

        if tokenizer:
            prompt_str = tokenizer.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_str = ""
        return prompt_str, current_messages

    def generate_correction_with_summary_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,
                                   prev_round_llm_raw_output,
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num,
                                   history_mode="all", summary=None):

        if history_mode == "all":
            current_messages = list(history_messages_from_prev_round)
        elif history_mode == "last_one":
            current_messages = [history_messages_from_prev_round[-1]]

        # Add PREVIOUS assistant's (failed) attempt
        assistant_content = prev_round_llm_raw_output
        current_messages.append({"role": "assistant", "content": assistant_content})

        # Add CURRENT user feedback and request for new attempt
        user_feedback_content = (
            f"The proof (Round {current_correction_round_num - 1}) is not correct. Following is the compilation error message, where we use <error></error> to signal the position of the error.\n\n{error_message_for_prev_round}"
            f"\nYou are provided with suggestions about how to fix the errors as follows:\n\n{summary}"
            "\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error message."
        )
        current_messages.append({"role": "user", "content": user_feedback_content})

        if tokenizer:
            prompt_str = tokenizer.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt_str = ""
        return prompt_str, current_messages


class DeepSeekNonCoTHandler(InferenceHandler):
    def __init__(self):
        pass 

    def prover_inference(self, lean4_code, tokenizer):
        formal_statement = lean4_code.split(":= by")[0] + ":= by" # don't include sorry, directly completion
        prompt = F"Complete the following Lean 4 code:\n\n```lean4\n{formal_statement}"
        return prompt, None

    def generate_correction_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,  # Not used by non-chat
                                   prev_round_llm_raw_output,  # Not used by non-chat directly in prompt
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num):
        original_stmt_for_completion = lean4_code_original_stmt# .split(":= by")[0] + ":= by"
        commented_errors = '\n'.join(
            [f'-- {line}' for line in error_message_for_prev_round.splitlines() if line.strip()])

        prompt_str = (
            f"-- The previous proof attempt (Round {current_correction_round_num - 1}) resulted in compilation errors:\n"
            f"{commented_errors}\n"
            f"-- Please provide a corrected version. Wrap the proof in ```lean4 and ```."
        )
        return prompt_str, None  # No message list

class KiminaCoTHandler(InferenceHandler):
    def __init__(self):
        pass 

    def prover_inference(self, lean4_code, tokenizer):
        formal_statement = lean4_code.split(":= by")[0] + ":= by"
        # don't include sorry https://huggingface.co/AI-MO/Kimina-Prover-Preview-Distill-7B
        problem = self.clean_code_string(formal_statement)
        prompt = "Think about and solve the following problem step by step in Lean 4."
        prompt += f"\n# Problem:{problem}"""
        prompt += f"\n# Formal statement:\n```lean4\n{formal_statement}\n```\n"

        messages = [
            {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return text, messages

    def generate_correction_prompt(self, lean4_code_original_stmt,
                                   history_messages_from_prev_round,
                                   prev_round_llm_raw_output,
                                   error_message_for_prev_round,
                                   tokenizer, current_correction_round_num):
        original_stmt_for_completion = lean4_code_original_stmt.split(":= by")[0] + ":= by"
        cleaned_original_problem_desc = self.clean_code_string(original_stmt_for_completion)
        current_messages = []

        current_messages = list(history_messages_from_prev_round)

        assistant_content = prev_round_llm_raw_output

        current_messages.append({"role": "assistant", "content": assistant_content})

        user_feedback_content = (
            f"The proof (Round {current_correction_round_num - 1}) is not correct. Following is the compilation error message, where we use <error></error> to signal the position of the error.\n\n{error_message_for_prev_round}"
            "\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error message."
        )
        current_messages.append({"role": "user", "content": user_feedback_content})

        prompt_str = tokenizer.apply_chat_template(current_messages, tokenize=False, add_generation_prompt=True)
        return prompt_str, current_messages

    def problem_check(self, statement, full_code):
        full_code = replace_statement_in_proof(statement, full_code)        
        return full_code

if __name__ == "__main__":

# Example multi-line string
    statement_string = """
import Mathlib
import Aesop
set_option maxHeartbeats 0
open BigOperators Real Nat Topology Rat

theorem lean_workbook_plus_34692_negation
  :¬( ∀
      (d₁ d₂ d₃ : ℝ)
      (h₀ : d₁ = 200)
      (h₁ : d₂ = 220)
      (h₂ : d₃ = 88),
    (d₁ + d₃) / (d₁ + d₂) * 100 = 68.57)
  := by sorry"""

    proof_string = """
import xxx
open xxx
set option xxx


lemma test lemma test_lemma_should_not_impact := by
    exact


def jjj hhh
    block

lemma test_v2 lemma test_lemma_should_not_impact_v23 := by
    sorry  
   
theorem to_proof_theorem_hh **this is the wrong condition** := by 
    exact
"""

    # Split into blocks
    print(replace_statement_in_proof(statement_string, proof_string))


