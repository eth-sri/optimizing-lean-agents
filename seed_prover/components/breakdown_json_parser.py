import os
import re
import json
from typing import Dict, List, Any
from loguru import logger
from jload import jsave
from core.component import Component
from prover.query.api import APIQuery
from utils import extract_lemma_dependencies
from metadata_utils import add_formalization, generate_uid, copy_metadata, add_lemma


class BreakdownJsonParserComponent(Component):
    """
    Component to parse breakdown output with JSON format inside <solution> tags.

    Expected input format:
        {
            "lemmas": [
                {
                    "id": <lemma number>,
                    "statement": <lemma statement>,
                    "assumption": <state the necessary assumptions for the lemma>,
                    "proof": <idea of the proof of the lemma in natural language>
                }
            ],
            "theorem": {
                "statement": <repeat the problem statement>,
                "proof": <idea of the proof, how to combine the lemmas into the final solution>
            }
        }
    """

    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)

    def parse_breakdown(self, content: str) -> Dict[str, Any]:
        """
        Parse the breakdown output into structured data.

        Args:
            content: String containing <solution> tags with JSON inside

        Returns:
            Dict with keys 'lemmas' (list), 'theorem' (dict), and 'full_breakdown' (str)
        """
        try:
            # Fix invalid escape sequences before parsing JSON
            # LLM might output backslashes that aren't valid JSON escapes (e.g., LaTeX \text)
            # Replace any backslash not followed by a valid JSON escape character with double backslash
            json_str = re.sub(r'(?<!\\)\\(?![\\/"bfnrtu])', r'\\\\', content)

            # Parse JSON
            data = json.loads(json_str)

            # Normalize lemma IDs to be sequential integers 1, 2, 3, ... n
            # This ensures we don't rely on the LLM to output correct IDs
            lemmas = data.get("lemmas", [])
            for idx, lemma in enumerate(lemmas, start=1):
                if lemma is not None:
                    lemma["id"] = idx

            result = {
                "lemmas": lemmas,
                "theorem": data.get("theorem", None),
                "full_breakdown": content
            }

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON content: {e}")
            logger.debug(f"Content: {content[:500]}")
            return {"lemmas": [], "theorem": None, "full_breakdown": content, "error": f"JSON parse error: {str(e)}"}
        except Exception as e:
            logger.error(f"Failed to parse breakdown content: {e}")
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

    def process(self, data_list: List[Dict[str, Any]], round_num: int = 0) -> List[Dict[str, Any]]:
        """
        Process a list of data items with breakdown results.

        Args:
            data_list: List of dictionaries containing 'informal_breakdown' key
            round_num: Current processing round

        Returns:
            List of dictionaries with added 'parsed_breakdown' key
        """
        logger.info(f"BreakdownParserComponent: Processing {len(data_list)} items, round {round_num}")

        # Create new items (one per sample)
        sampled_items = []

        for item in data_list:
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
        output_dir = os.path.join(base_output_dir, f"round{round_num}", "breakdown_parser")
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
