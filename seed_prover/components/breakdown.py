import os
import random
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from core.component import Component
from jload import jload, jsave
from prover.query.api import APIQuery
from loguru import logger
from metadata_utils import add_breakdown, generate_uid, copy_metadata


class BreakdownComponent(Component):

    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)
        
        self.base_template_path = f"seed_prover/template/breakdown/{self.config.get('template')}.md"
        self.base_template = self._load_template(self.base_template_path)
        self.feedback_template_path = f"seed_prover/template/breakdown/feedback_prover.md"
        self.feedback_template = self._load_template(self.feedback_template_path)
        

    def process(self, data_list, round_num=0):
        """
        Break down problems into smaller parts.
        
        Args:
            data_list: List of problem dictionaries
            round_num: Current correction round (0 = initial)
            
        Returns:
            List[Dict]: Problems with added breakdown results
        """
        print(f"BreakdownComponent: Processing {len(data_list)} problems, round {round_num}")
        
        data_for_processing = self._preprocess(data_list, round_num)
        prompts = self._prepare_prompt(data_for_processing)

        for i, response, detailed_cost in self.querier.run_queries(prompts):
            # Strip thinking from breakdown to reduce token usage in downstream components
            if '</think>' in response:
                thinking_part, content_part = response.split('</think>', 1)
                data_for_processing[i]["informal_breakdown"] = content_part.strip()
                data_for_processing[i]["informal_breakdown_reasoning"] = thinking_part.strip()
            else:
                data_for_processing[i]["informal_breakdown"] = response
            data_for_processing[i]["detailed_cost"] = detailed_cost
            
        base_output_dir = self.global_config.get("output_dir")
        output_dir = os.path.join(base_output_dir, f"round{round_num}", "breakdown")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "breakdown.json")
        jsave(data_for_processing, output_file)

        return data_for_processing

    def _preprocess(self, data_list, round_num=0):

        data_for_processing = []
        # Use recursive_sample_times for round 1+, otherwise use sample_times
        if round_num > 0 and "recursive_sample_times" in self.config:
            sample_times = self.config.get("recursive_sample_times")
            logger.info(f"Round {round_num}: Using recursive_sample_times={sample_times}")
        else:
            sample_times = self.config.get("sample_times")
            logger.info(f"Round {round_num}: Using sample_times={sample_times}")

        for item in data_list:
            # Get metadata from item, should already be initialized
            metadata = item.get("metadata")
            if not metadata:
                raise ValueError(f"Item missing metadata field: {item}")

            for n in range(sample_times):
                item_copy = item.copy()

                # Add breakdown_id to metadata
                new_metadata = add_breakdown(metadata, n)
                item_copy["metadata"] = new_metadata

                # Generate uid from metadata
                item_copy["uid"] = generate_uid(new_metadata)

                data_for_processing.append(item_copy)

        return data_for_processing

    def _load_template(self, template_path):

        if not template_path or not os.path.exists(template_path):
            raise ValueError(f"Template path {template_path} does not exist or is not set in config")

        with open(template_path, "r") as f:
            template = f.read()

        return template

    def _prepare_prompt(self, data_list):
        """
        Prepare the prompt for the LLM based on the problem.

        Args:
            data_list: List of problem dictionaries

        Returns:
            str: Formatted prompt
        """
        prompts = []
        for problem in data_list:
            metadata = problem.get("metadata", {})
            # if origin_problem_id != parent_problem_id, we are in feedback mode, otherwise in round 0
            if metadata.get("parent_problem_id") != metadata.get("origin_problem_id"):
                template = self.feedback_template
            else:
                template = self.base_template

            # Add num_lemmas instruction if specified
            num_lemmas = self.config.get("num_lemmas")
            if num_lemmas and isinstance(num_lemmas, (int, float)):
                # Split template into lines
                template_lines = template.split('\n')
                # Insert the instruction after line 1 (index 0)
                if len(template_lines) > 0:
                    instruction = f"Break the problem down into {int(num_lemmas)} lemmas that are of similar difficulty."
                    template_lines.insert(1, instruction)
                    template = '\n'.join(template_lines)

            prompt = template.format(**problem)
            prompts.append([
                {
                    "role": "user",
                    "content": prompt
                }
            ])
            problem["breakdown_prompt"] = prompt
        return prompts

    def _query_llm(self, prompts):
        """
        Query the LLM with the prepared prompts.
        
        Args:
            prompts: List of formatted prompts
            
        Returns:
            List[str]: Responses from the LLM
        """
        if self.config.get("api_provider") == "vllm_offline":
            return self._vllm_inference(prompts)
        else:
            responses = [""] * len(prompts)
            for i, result in self.api_query.run_queries(prompts):
                responses[i] = result
                print(result)
            return responses

    def _vllm_inference(self, prompts):
        from vllm import LLM, SamplingParams

        api_config = self.config.get("api")
        model_path = api_config.get("model")
        max_model_len = api_config.get("max_tokens", 20000)
        temperature = api_config.get("temperature", 1.0)
        gpu = self.config.get("gpu")
        node = self.config.get("node")

        logger.info(f"Using VLLM offline inference with model: {model_path}, max tokens: {max_model_len}, temperature: {temperature}, gpu: {gpu}, node: {node}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processed_prompts = [
            tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) 
            for prompt in prompts
        ]
        if node > 1:
            model = LLM(model=model_path, trust_remote_code=True, 
                        max_model_len=max_model_len, tensor_parallel_size=gpu, 
                        pipeline_parallel_size=node, distributed_executor_backend="ray")
        else:
            model = LLM(model=model_path, trust_remote_code=True, 
                        max_model_len=max_model_len, tensor_parallel_size=gpu)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_model_len,
            top_p=0.95,
            n=1,
        )

        outputs = model.generate(
            processed_prompts,
            sampling_params=sampling_params
        )

        responses = [
            output.outputs[0].text.strip() if output.outputs else ""
            for output in outputs
        ]

        return responses

    def _post_process(self, responses, data_list, round_num=0):
        """
        Post-process the LLM responses and update the data list.
        
        Args:
            responses: List of responses from the LLM
            data_list: Original list of problem dictionaries
            
        Returns:
            List[Dict]: Updated problem dictionaries with breakdown results
        """
        for i, response in enumerate(responses):
            # Strip thinking from breakdown to reduce token usage in downstream components
            if '</think>' in response:
                thinking_part, content_part = response.split('</think>', 1)
                data_list[i]["informal_breakdown"] = content_part.strip()
                data_list[i]["informal_breakdown_reasoning"] = thinking_part.strip()
            else:
                data_list[i]["informal_breakdown"] = response

        base_output_dir = self.global_config.get("output_dir")
        output_dir = os.path.join(base_output_dir, f"round{round_num}", "breakdown")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "breakdown.json")
        jsave(data_list, output_file)

        return data_list