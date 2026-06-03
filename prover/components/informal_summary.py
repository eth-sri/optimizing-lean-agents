import os
import json
import random
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from core.component import Component
from prover.utils import DeepSeekCoTHandler, DeepSeekNonCoTHandler, KiminaCoTHandler, get_error_str
from jload import jload, jsave
from prover.query.api import APIQuery


def handle(text):
    """Filter out import, set_option, and open statements from Lean code."""
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not (
            line.strip().startswith('import') or
            line.strip().startswith('set_option') or
            line.strip().startswith('open')
    )]
    return '\n'.join(filtered_lines)


class InformalSummaryComponent(Component):
    def process(self, data_list, round_num=0):
        print(round_num)
        data_output = self._informal_summarize(data_list, round_num)

        print(f"InformalSummaryComponent: Processing {len(data_output)} problems, round {round_num}")

        return data_output

    def _informal_summarize(self, data_list, correction_round_num):
        output_dir = self.global_config.get('output_dir')
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

        querier = APIQuery(
            model=model,
            api=api,
            **kwargs
        )

        template_path = self.config.get('informal_template_path')

        template = self._load_template(template_path)

        prompts = []
        records = []
        if api == "vllm":
            tokenizer = AutoTokenizer.from_pretrained(model)
        else:
            tokenizer = None
        
        for item in data_list:
            error_str = get_error_str(
                        item.get('compiled_code_that_failed_in_prev_round', ''),
                        item.get('errors_for_compiled_code_from_prev_round', {}).get('errors', []),
                        False
                    )
            item["error_info"] = error_str
            prompt = template.format(**item)
            message = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            if api == "vllm":
                prompt_str = tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True)
                num_tokens = len(tokenizer.tokenize(prompt_str))
            else:
                prompt_str = ""
                num_tokens = 0
            records.append(
                {
                    "token_nums": num_tokens,
                    "prompts": message,
                    "messages_lists_for_current_prompts": message,
                    "input_items": item
                }
            )
        
        df_med = pd.DataFrame(records)
        max_length = max_tokens * 3 / 4  # fixed to be Qwen
        to_process_df = df_med[df_med.token_nums <= max_length].reset_index(drop=True)
        print(f"In total {len(df_med)}, selected {len(to_process_df)} whose length is smaller than {max_length}")

        summary_data = []
        data_list_with_summary = []
        for i, response, detailed_cost in querier.run_queries(to_process_df.prompts):
            item = to_process_df.input_items[i]
            item["informal_summary"] = response
            data_list_with_summary.append(item)

            # TODO: handle reasoning in the response
            think = ""
            response_text = response

            summary_data.append({
                "problem_id": item["problem_id"],
                "origin_problem_id": item.get("origin_problem_id"),
                "id_maps": item.get("id_maps"),
                "lean4_code": item["lean4_code"],
                "full_code": item["compiled_code_that_failed_in_prev_round"],
                "error_info": item["errors_for_compiled_code_from_prev_round"],
                "informal_summary_output": response,
                "informal_summary_think": think,
                "informal_summary_response": response_text,
                "detailed_cost": detailed_cost,
            })            

        informal_summary_file = os.path.join(output_dir, f'informal_summary_round_{correction_round_num}.json')
        jsave(summary_data, informal_summary_file)
        print(f"Informal summaries saved to {informal_summary_file}")

        return data_list_with_summary

    def _load_template(self, template_path):

        if not template_path or not os.path.exists(template_path):
            raise ValueError(f"Template path {template_path} does not exist or is not set in config")

        with open(template_path, "r") as f:
            template = f.read()

        return template