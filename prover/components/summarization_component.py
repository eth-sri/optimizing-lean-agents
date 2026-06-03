import os
import pandas as pd
import numpy as np
from prover.core.component import Component
from id_utils import get_breakdown_id, get_lemma_component, get_lemma_id
from metadata_utils import (
    get_origin_problem_id,
    create_problem_key,
    create_breakdown_key,
    create_lemma_key,
    create_attempt_key
)


class SummarizationComponent(Component):
    """
    Component to generate summary reports of pipeline results.
    """

    def __init__(self, name, component_config, global_config):
        super().__init__(name, component_config, global_config)
        # use_metadata controls whether to use metadata/uid based IDs (new) or problem_id (legacy)
        # Default: False for backwards compatibility
        self.use_metadata = self.config.get('use_metadata', False)

    def process(self, data_list, round_num=0):
        """
        Generate summary reports from compilation results.

        Args:
            data_list: List of problem dictionaries with all results
            output_dir: Output directory path
            round_num: Final round number

        Returns:
            List[Dict]: Same data (summarization is mostly side effects)
        """
        verbosity = self.global_config.get('verbosity', 3)

        if verbosity >= 1:
            print(f"SummarizationComponent:")
            print(f"  Field: {self.config.get('field', 'complete')}")

        # Get configuration
        output_dir = self.global_config.get('output_dir')
        summary_output_dir = os.path.join(output_dir, f'summary_round_{round_num}')
        os.makedirs(summary_output_dir, exist_ok=True)
        field = self.config.get('field', 'complete')
        
        try:
            # Load compilation results file to get the full compilation data
            file_suffix = f"_corr{round_num}" if round_num > 0 else ""
            compilation_file = os.path.join(output_dir, f'code_compilation_repl{file_suffix}.json')
            full_record_file = os.path.join(output_dir, f'full_records{file_suffix}.json')
            
            if not os.path.exists(compilation_file):
                print(f"Compilation results file not found: {compilation_file}")
                return data_list
            
            if not os.path.exists(full_record_file):
                print(f"Full records file not found: {full_record_file}")
                return data_list
            
            # Load data files
            df = pd.read_json(compilation_file)
            df_full = pd.read_json(full_record_file)

            if verbosity >= 1:
                print(f"Loaded {len(df)} compilation results and {len(df_full)} full records")

            if df.empty or df_full.empty:
                print("Empty data files, skipping summarization")
                return data_list

            # Check if we should use metadata-based grouping
            if self.use_metadata:
                # NEW FORMAT: Use metadata-based grouping with simple string keys
                if verbosity >= 2:
                    print("Using metadata-based grouping")

                # Check if data has metadata
                if 'metadata' not in df_full.columns or df_full['metadata'].isna().all():
                    print("Error: use_metadata=True but no metadata found in data")
                    return data_list

                # Create hierarchical keys from metadata using utility functions
                df_full['problem_key'] = df_full['metadata'].apply(create_problem_key)
                df_full['breakdown_key'] = df_full['metadata'].apply(create_breakdown_key)
                df_full['lemma_key'] = df_full['metadata'].apply(create_lemma_key)
                df_full['attempt_key'] = df_full['metadata'].apply(create_attempt_key)

                # Use uid as the ID key for joining
                if 'uid' not in df_full.columns:
                    print("Error: use_metadata=True but no uid found in data")
                    return data_list

                id_key = 'uid'

            else:
                # LEGACY FORMAT: Use old id_maps format
                if verbosity >= 2:
                    print("Using legacy id_maps format")

                # Check if data has id_maps
                if 'id_maps' not in df_full.columns or df_full['id_maps'].isna().all():
                    print("Error: use_metadata=False but no id_maps found in data")
                    return data_list

                id_key = 'problem_id'

            # Calculate correctness based on the specified field
            df["correct"] = df.apply(lambda row: int(
                ((row["compilation_result"][field])) and
                ("apply?" not in row["code"]) and
                ("exact?" not in row["code"])
            ), axis=1)

            if self.use_metadata:
                # Add keys to compilation df by looking up from df_full
                key_lookups = {
                    'problem_key': dict(zip(df_full[id_key], df_full['problem_key'])),
                    'breakdown_key': dict(zip(df_full[id_key], df_full['breakdown_key'])),
                    'lemma_key': dict(zip(df_full[id_key], df_full['lemma_key'])),
                    'attempt_key': dict(zip(df_full[id_key], df_full['attempt_key']))
                }

                for key_name, lookup in key_lookups.items():
                    df[key_name] = df['name'].apply(lambda x: lookup.get(x, 'unknown'))

                # Generate summaries for each hierarchical level
                meta_result = []
                summary_levels = [
                    ('problem_key', 'origin_problem_id'),
                    ('breakdown_key', 'breakdown_id'),
                    ('lemma_key', 'lemma_id'),
                    ('attempt_key', 'attempt_id')
                ]

                for key_col, level_name in summary_levels:
                    # Group by key and calculate statistics
                    df_grp = df[[key_col, "correct"]].groupby(key_col)["correct"].aggregate(["sum", "count"]).reset_index()

                    # Save detailed CSV
                    csv_path = os.path.join(summary_output_dir, f"{level_name}_summarize.csv")
                    df_grp.to_csv(csv_path, index=False, header=True, sep='\t', quoting=1, na_rep='Missing')
                    if verbosity >= 2:
                        print(f"Saved detailed summary: {csv_path}")

                    # Calculate meta statistics
                    problem_num = len(df_grp)
                    solved_num = sum(df_grp["sum"] > 0)
                    solved_ratio = f"{solved_num / problem_num * 100:.2f}" if problem_num > 0 else "0.00"

                    meta_result.append({
                        "level": level_name,
                        "value": {
                            "problem_num": problem_num,
                            "solved_num": solved_num,
                            "solved_ratio": solved_ratio
                        }
                    })

                    print(f"Level {level_name}: {solved_num}/{problem_num} problems solved ({solved_ratio}%)")

            else:
                # LEGACY: Use old id_maps format
                ids_lookup = dict(zip(df_full[id_key], df_full.id_maps))

                # Determine the number of ID levels
                ids_num_ = np.unique(df_full.id_maps.apply(lambda x: len(x)))
                if len(ids_num_) != 1:
                    print(f"Warning: Inconsistent id_maps lengths: {ids_num_}")
                    return data_list

                ids_num = ids_num_[0]
                if df_full.empty or len(df_full.id_maps) == 0:
                    print("No id_maps available for summarization")
                    return data_list

                first_element = df_full.id_maps.iloc[0]

                # Generate summaries for each ID level
                meta_result = []

                for i in range(ids_num):
                    names = [k for k, _ in first_element[i].items()]
                    if len(names) != 1:
                        print(f"Warning: Expected 1 name at level {i}, got {len(names)}: {names}")
                        continue

                    name = names[0]

                    # Add the ID field to dataframe
                    df[name] = df["name"].apply(lambda x: ids_lookup[x][i][name] if x in ids_lookup else x)

                    # Group by the ID field and calculate statistics
                    df_grp = df[[name, "correct"]].groupby(name)["correct"].aggregate(["sum", "count"]).reset_index()

                    # Save detailed CSV
                    csv_path = os.path.join(summary_output_dir, f"{name}_summarize.csv")
                    df_grp.to_csv(csv_path, index=False, header=True, sep='\t', quoting=1, na_rep='Missing')
                    if verbosity >= 2:
                        print(f"Saved detailed summary: {csv_path}")

                    # Calculate meta statistics
                    problem_num = len(df_grp)
                    solved_num = sum(df_grp["sum"] > 0)
                    solved_ratio = f"{solved_num / problem_num * 100:.2f}" if problem_num > 0 else "0.00"

                    meta_result.append({
                        "level": f"{name}",
                        "value": {
                            "problem_num": problem_num,
                            "solved_num": solved_num,
                            "solved_ratio": solved_ratio
                        }
                    })

                    print(f"Level {name}: {solved_num}/{problem_num} problems solved ({solved_ratio}%)")
            
            # Save meta summary
            meta_df = pd.DataFrame(meta_result)
            meta_path = os.path.join(summary_output_dir, "meta_summarize.json")
            meta_df.to_json(meta_path, indent=4, orient="records")
            if verbosity >= 2:
                print(f"Saved meta summary: {meta_path}")

            # Minify results if configured
            if self.config.get('minify_results', False):
                self._minify_results(output_dir, df, df_full, field, verbosity)

            if verbosity >= 1:
                print("Summarization completed successfully")
            return data_list

        except Exception as e:
            print(f"Error during summarization: {e}")
            import traceback
            traceback.print_exc()
            return data_list

    def _minify_results(self, output_dir, df, df_full, field, verbosity):
        """Create minified proof_attempts.json compatible with proof_simulation loader."""
        import json

        minified_dir = os.path.join(output_dir, "minified")
        os.makedirs(minified_dir, exist_ok=True)

        # Build compilation lookup: uid/problem_id -> compilation result
        compilation_lookup = {}
        for _, row in df.iterrows():
            compilation_lookup[row["name"]] = row["compilation_result"]

        # Load to_inference_codes for detailed_cost and full_code
        inference_lookup = {}
        for suffix in ["", "_corr1"]:
            inference_file = os.path.join(output_dir, f"to_inference_codes{suffix}.json")
            if os.path.exists(inference_file):
                with open(inference_file) as f:
                    for record in json.load(f):
                        key = record.get("uid", record.get("problem_id", ""))
                        inference_lookup[key] = record

        proof_attempts = []

        for _, row in df_full.iterrows():
            uid = row.get("uid", row.get("problem_id", row.get("name", "unknown")))
            comp = compilation_lookup.get(uid, {})
            inf = inference_lookup.get(uid, {})
            meta = row.get("metadata", {})

            passed = comp.get("pass", False) if isinstance(comp, dict) else False
            complete = comp.get("complete", False) if isinstance(comp, dict) else False

            entry = {
                "origin_problem_id": meta.get("origin_problem_id", row.get("origin_problem_id", uid)),
                "attempt_id": meta.get("attempt_id", row.get("attempt_id", 0)),
                "correction_round_id": meta.get("correction_round_id", row.get("correction_round_id", 0)),
                "pass": passed,
                "complete": complete,
                "detailed_cost": inf.get("detailed_cost", {}),
                "model_config_path": inf.get("model_config_path", row.get("model_config_path")),
                "full_code": inf.get("full_code", row.get("full_code")),
            }

            proof_attempts.append(entry)

        with open(os.path.join(minified_dir, "proof_attempts.json"), "w") as f:
            json.dump(proof_attempts, f, indent=2)

        if verbosity >= 1:
            print(f"Minified results saved to {minified_dir} ({len(proof_attempts)} attempts)")
