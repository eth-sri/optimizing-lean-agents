import os
import random
import json
from typing import Dict, Any
from pydantic import BaseModel, Field
from typing import List
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
from core.component import Component
from jload import jload, jsave
from prover.query.api import APIQuery
from loguru import logger
from metadata_utils import add_breakdown, generate_uid, copy_metadata
from utils import extract_lemma_dependencies


class Lemma(BaseModel):
    id: int = Field(..., description="The unique number identifier for the lemma (e.g., 1, 2).")
    statement: str = Field(..., description="The mathematical statement of the lemma.")
    assumption: str = Field(..., description="The necessary assumptions required for this lemma to hold.")
    proof: str = Field(..., description="The idea of the proof in natural language. References other lemmas if necessary.")

class Theorem(BaseModel):
    statement: str = Field(..., description="Restatement of the main problem or theorem to be solved.")
    proof: str = Field(..., description="The final proof idea demonstrating how to combine the lemmas into the solution.")

class BreakdownSchema(BaseModel):
    lemmas: List[Lemma] = Field(..., description="A list of supporting lemmas required to prove the theorem.")
    theorem: Theorem = Field(..., description="The main theorem and its synthesis proof.")

class BreakdownJsonComponent(Component):

    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)
        self.base_template_path = f"seed_prover/template/breakdown_json/{self.config.get('template')}.md"
        self.base_template = self._load_template(self.base_template_path)
        self.feedback_template_path = f"seed_prover/template/breakdown_json/feedback_prover.md"
        self.feedback_template = self._load_template(self.feedback_template_path)

        # Load retrieval section template (optional - for use with retrieval component)
        # self.retrieval_section_template_path = "seed_prover/template/retrieval/useful_theorems_section.md"
        # self.retrieval_section_template = self._load_template_optional(self.retrieval_section_template_path)

        self.output_schema = BreakdownSchema.model_json_schema()
        

    def parse_breakdown(self, content: str) -> Dict[str, Any]:
        """Parse JSON breakdown from LLM response."""
        try:
            data = json.loads(content)
            # Normalize lemma IDs to be sequential
            lemmas = data.get("lemmas", [])
            for idx, lemma in enumerate(lemmas, start=1):
                if lemma is not None:
                    lemma["id"] = idx
            return {
                "lemmas": lemmas,
                "theorem": data.get("theorem", None),
                "full_breakdown": content
            }
        except Exception as e:
            logger.error(f"Failed to parse breakdown: {e}")
            return {"lemmas": [], "theorem": None, "full_breakdown": content, "error": str(e)}

    def _add_dependencies_to_breakdown(self, parsed_breakdown: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add dependency information to lemmas and theorem in parsed breakdown.

        Dependencies are stored as simple lemma_id integers. The full context
        (origin_problem_id, breakdown_id, etc.) is inferred from the item's metadata.

        Args:
            parsed_breakdown: The parsed breakdown dict with lemmas and theorem
            metadata: The metadata dict (used for context validation)

        Returns:
            Updated parsed_breakdown with dependencies added
        """
        lemmas = parsed_breakdown.get("lemmas", [])

        # Extract dependencies for each lemma
        for lemma in lemmas:
            lemma_id = lemma.get("id")

            # Combine all text fields to search for dependencies
            combined_text = ""
            if lemma.get('statement'):
                combined_text += str(lemma.get('statement', '')) + "\n"
            if lemma.get('assumption'):
                combined_text += str(lemma.get('assumption', '')) + "\n"
            if lemma.get('proof'):
                combined_text += str(lemma.get('proof', '')) + "\n"

            # Extract referenced lemma numbers (just store the integers)
            dep_lemma_ids = extract_lemma_dependencies(combined_text)

            # Add dependencies as integers (context is inferred from metadata)
            lemma["dependencies"] = dep_lemma_ids

        # Theorem depends on all lemmas in the breakdown
        theorem = parsed_breakdown.get("theorem")
        if theorem:
            # Store just the lemma IDs (context inferred from metadata)
            all_lemma_ids = [lemma.get('id') for lemma in lemmas]
            theorem["dependencies"] = all_lemma_ids

        return parsed_breakdown

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
                data_for_processing[i]["informal_breakdown"] = content_part.strip().strip('```json').strip('```').strip()
                data_for_processing[i]["informal_breakdown_reasoning"] = thinking_part.strip()
            else:
                data_for_processing[i]["informal_breakdown"] = response
            data_for_processing[i]["detailed_cost"] = detailed_cost

        # Create new items (one per sample)
        sampled_items = []

        for item in data_for_processing:
            # idx = result["index"]
            # meta = prompt_metadata[idx]
            # item_idx = meta["item_idx"]
            # sample_idx = meta["sample_idx"]

            # # Get the original problem
            # original_item = data_list[item_idx]
            # # Get metadata from original item
            # original_metadata = original_item.get("metadata")
            # if not original_metadata:
            #     raise ValueError(f"Item missing metadata field: {original_item}")

            # Create a new item for this sample (copy all fields from original)
            sample_item = item.copy()
            # else: keep original metadata

            # Add sample metadata for tracking
            sample_item["sample_idx"] = 1

            # Add parsing results
            sample_item["structured_breakdown"] = item["informal_breakdown"]

            # Parse the structured response
            parsed = self.parse_breakdown(item["informal_breakdown"])
            sample_item["parsed_breakdown"] = parsed

            sampled_items.append(sample_item)

        # Add dependencies to successfully parsed breakdowns
        for item in sampled_items:
            parsed_breakdown = item.get("parsed_breakdown", {})
            if "error" not in parsed_breakdown and parsed_breakdown.get("lemmas") is not None:
                item_metadata = item.get("metadata")
                if not item_metadata:
                    raise ValueError(f"Item missing metadata: {item.get('uid', 'unknown')}")
                parsed_breakdown = self._add_dependencies_to_breakdown(
                    parsed_breakdown,
                    item_metadata
                )
                # print("Parsed breakdown with dependencies:", parsed_breakdown)
                item["parsed_breakdown"] = parsed_breakdown

        # Filter out failed parses
        successful_items = []
        failed_items = []

        for item in sampled_items:
            parsed_breakdown = item.get("parsed_breakdown", {})
            if "error" in parsed_breakdown:
                failed_items.append(item)
                logger.warning(f"Failed to parse breakdown for {item.get('problem_id', 'unknown')}: {parsed_breakdown.get('error')}")
            elif not parsed_breakdown.get("theorem"):
                failed_items.append(item)
                logger.warning(f"No theorem found in parsed breakdown for {item.get('problem_id', 'unknown')}")
            else:
                successful_items.append(item)

        # Log statistics
        logger.info(f"Breakdown parsing complete: {len(successful_items)}/{len(sampled_items)} successful")

        if failed_items:
            logger.warning(f"Failed to parse {len(failed_items)} breakdowns")

        # Save results
        base_output_dir = self.global_config.get("output_dir")
        output_dir = os.path.join(base_output_dir, f"round{round_num}", "breakdown")
        os.makedirs(output_dir, exist_ok=True)

        # Save only successful items to parsed_breakdown.json
        output_file = os.path.join(output_dir, "parsed_breakdown.json")
        jsave(successful_items, output_file)

        # Save failed items separately for analysis
        if failed_items:
            failed_file = os.path.join(output_dir, "failed_parses.json")
            jsave(failed_items, failed_file)
            logger.info(f"Saved {len(failed_items)} failed parses to {failed_file}")

        # Return only successful items (those with valid parsed breakdowns)
        # Failed parses are saved separately for analysis but not passed to formalization
        return successful_items

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
