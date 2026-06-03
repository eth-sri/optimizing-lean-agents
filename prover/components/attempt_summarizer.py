import os
import json
import re
import sys
from pathlib import Path
import yaml
from loguru import logger
from jload import jload, jsave

try:
    # Add prover directory to path to find query module
    prover_dir = Path(__file__).parent.parent
    if str(prover_dir) not in sys.path:
        sys.path.insert(0, str(prover_dir))
    from prover.query.api import APIQuery
except ImportError:
    APIQuery = None

try:
    from prover.core.component import Component
except ImportError:
    Component = object  # Fallback for when prover.core is not available


# Error classification patterns from scripts/error_statistics/error_classify.py
ERROR_PATTERNS = {
    "tactic_failed": r"\btactic(?:\s+['''](?P<what>[^''']+)['''])?\s+failed\b",
    "mismatch": r"\bmismatch\b",
    "failed_to_synthesize": r"\bfailed to synthesize\b",
    "unexpected": r"\bunexpected\b",
    "expected": r"\bexpected\b",
    "linarith_failed": r"\blinarith failed to find a contradiction\b",
    "max_recursion_depth": r"\bmaximum recursion depth\b",
    "omega_failed": r"\bomega could not prove the goal\b",
    "simp_no_progress": r"\bsimp made no progress\b",
    "simp_all_no_progress": r"\bsimp_all made no progress\b",
    "made_no_progress": r"\bmade no progress\b",
    "unknown": r"\bunknown\b",
    "unsolved_goals": r"\bunsolved goals\b",
    "invalid": r"\binvalid\b",
    "already_declared": r"\bhas already been declared\b"
}


class AttemptSummarizerComponent(Component):
    """
    Summarizes proof attempts with two modes:
    1. reasoning_trace: LLM-based summarization of reasoning traces
    2. compilation: Classification of compilation errors by type
    """

    @classmethod
    def from_config_file(cls, config_path, output_dir):
        """
        Create an AttemptSummarizerComponent from a standalone config file.

        Useful for on-the-fly summarization in DataLoader without needing
        a full global_config object.

        Args:
            config_path: Path to the summarization config YAML file
            output_dir: Base output directory where full_records and compilation files are located

        Returns:
            AttemptSummarizerComponent instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract summarization config
        summarization_config = config.get('summarization', {})

        # Create minimal global_config for compatibility
        global_config = {
            'output_dir': output_dir
        }

        # Create instance with name 'on_the_fly_summarizer'
        return cls(
            name='on_the_fly_summarizer',
            component_config=summarization_config,
            global_config=global_config
        )

    def __init__(self, name, component_config, global_config):
        # Don't call super().__init__ with model_config since we handle it differently
        self.name = name
        self.config = component_config
        self.global_config = global_config

        # Initialize reasoning trace summarizer if enabled
        self.reasoning_trace_enabled = False
        self.querier = None
        self.override_existing_summaries = self.config.get('override_existing_summaries', False)

        # Initialize attempt summaries file path
        output_dir = global_config.get('output_dir')
        self.attempt_summary_dir = os.path.join(output_dir, 'attempt_summary')
        self.attempt_summaries_file = os.path.join(self.attempt_summary_dir, 'attempt_summaries.json')
        self.failed_summaries_file = os.path.join(self.attempt_summary_dir, 'failed_summaries.json')

        reasoning_config = self.config.get('reasoning_trace', {})
        if reasoning_config.get('enabled', False):
            self.reasoning_trace_enabled = True

            # Load model config
            model_config_path = reasoning_config.get('model_config')
            if not model_config_path:
                raise ValueError("reasoning_trace requires model_config when enabled")

            with open(model_config_path, 'r') as f:
                model_config = yaml.safe_load(f)

            # Load template
            template_config = reasoning_config.get('template', 'reasoning_summarizer')

            # Check if template_config is an absolute path (for on-the-fly summarization)
            if template_config.startswith('/') or '\\' in template_config or ':\\' in template_config:
                # Absolute path provided
                self.template_path = template_config
            else:
                # Relative template name - construct path
                self.template_path = f"prover/template/reasoning_summarizer/{template_config}.md"

            self.template = self._load_template(self.template_path)

            # Initialize API querier
            if not APIQuery:
                raise ImportError("APIQuery not available. Please ensure query package is installed.")

            model = model_config.get('model')
            api = model_config.get('api')
            max_tokens = model_config.get('max_tokens', 8192)
            temperature = model_config.get('temperature', 0.7)

            kwargs = model_config.copy()
            for key in ['model', 'api', 'date']:
                if key in kwargs:
                    del kwargs[key]
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

            self.querier = APIQuery(
                model=model,
                api=api,
                **kwargs
            )

        # Check if compilation error classification is enabled
        compilation_config = self.config.get('compilation', {})
        self.compilation_enabled = compilation_config.get('enabled', False)

    def process(self, data_list, round_num=0):
        """
        Summarize proof attempts by reading from full_records and code_compilation files.

        Args:
            data_list: List of problem dictionaries (passed through unchanged)
            round_num: Current correction round

        Returns:
            List[Dict]: Original data_list unchanged
        """
        output_dir = self.global_config.get('output_dir')

        # Determine file paths
        if round_num == 0:
            records_file = os.path.join(output_dir, 'full_records.json')
            compilation_file = os.path.join(output_dir, 'code_compilation_repl.json')
        else:
            records_file = os.path.join(output_dir, f'full_records_corr{round_num}.json')
            compilation_file = os.path.join(output_dir, f'code_compilation_repl_corr{round_num}.json')

        # Process full_records for reasoning trace summarization
        if self.reasoning_trace_enabled and os.path.exists(records_file):
            records = jload(records_file)
            if records:
                logger.info(f"Loaded {len(records)} records from {records_file}")
                records = self._summarize_reasoning_traces(records)
                jsave(records, records_file)
                logger.info(f"Saved reasoning summaries to {records_file}")

        # Process code_compilation for compilation error classification
        if self.compilation_enabled and os.path.exists(compilation_file):
            compilation_records = jload(compilation_file)
            if compilation_records:
                logger.info(f"Loaded {len(compilation_records)} compilation records from {compilation_file}")
                compilation_records = self._classify_compilation_errors(compilation_records)
                jsave(compilation_records, compilation_file)
                logger.info(f"Saved compilation summaries to {compilation_file}")

        # Return original pipeline data unchanged
        return data_list

    def _summarize_reasoning_traces(self, data_list, clear_files=True):
        """Summarize reasoning traces using LLM."""
        # Empty the attempt_summaries.json and failed_summaries.json files at the start
        if clear_files:
            self._clear_attempt_summaries_file()
            self._clear_failed_summaries_file()

        # Filter records that need summarization
        records_to_summarize = []
        indices_to_summarize = []

        for i, record in enumerate(data_list):
            if record.get('model_output') and (not record.get('reasoning_summary') or self.override_existing_summaries):
                records_to_summarize.append(record)
                indices_to_summarize.append(i)

        if not records_to_summarize:
            logger.info("All records already have reasoning summaries or no model_output")
            return data_list

        logger.info(f"Summarizing {len(records_to_summarize)} reasoning traces")

        # Prepare prompts
        prompts = self._prepare_prompts(records_to_summarize)

        # Track validation statistics
        validation_stats = {
            'total_processed': 0,
            'valid': 0,
            'invalid': 0,
            'missing_fields': [],
            'parse_errors': 0
        }

        # Run LLM queries
        for i, response, detailed_cost in self.querier.run_queries(prompts):
            original_idx = indices_to_summarize[i]

            # Parse JSON response
            summary = self._parse_summary_response(response)

            # Add proof length (number of lines) from the record's code output
            record = records_to_summarize[i]
            proof_code = record.get('full_code') or record.get('lean4_code', '')
            summary['proof_length'] = len([line for line in proof_code.splitlines() if line.strip()]) if proof_code else 0

            # Count number of ":= by sorry" occurrences
            summary['sorry_count'] = proof_code.count(':= by sorry') if proof_code else 0

            # Validate the parsed summary
            validation_result = self._validate_summary_fields(summary, original_idx)
            summary['validation'] = {
                'is_valid': validation_result['is_valid'],
                'missing_fields': validation_result['missing_fields'],
                'has_parse_error': validation_result.get('has_parse_error', False)
            }

            # Update statistics
            validation_stats['total_processed'] += 1
            if validation_result['is_valid']:
                validation_stats['valid'] += 1
            else:
                validation_stats['invalid'] += 1
                validation_stats['missing_fields'].extend(validation_result['missing_fields'])

            if validation_result.get('has_parse_error'):
                validation_stats['parse_errors'] += 1

            data_list[original_idx]['reasoning_summary'] = summary
            data_list[original_idx]['reasoning_summary_cost'] = detailed_cost

            # Save full raw response to attempt_summaries.json with metadata
            record = records_to_summarize[i]
            summary_record = {
                'record_index': original_idx,
                'uid': record.get('uid', 'unknown'),
                'name': record.get('name', 'unknown'),
                'parsed_summary': summary,
                'validation': summary.get('validation', {}),
                'detailed_cost': detailed_cost,
                'raw_response': response
            }
            self._append_attempt_summary(summary_record)

            # If validation failed, also save to failed_summaries.json for easier debugging
            if not validation_result['is_valid']:
                self._append_failed_summary(summary_record)

        # Log validation statistics
        self._log_validation_stats(validation_stats, 'reasoning_trace')
        logger.info(f"Completed reasoning trace summarization with validation")

        return data_list

    def _classify_compilation_errors(self, data_list):
        """Classify compilation errors by type."""
        classified_count = 0
        validation_stats = {
            'total_processed': 0,
            'valid': 0,
            'invalid': 0,
            'missing_fields': [],
            'success': 0,
            'no_errors_found': 0,
            'has_errors': 0
        }

        for record in data_list:
            # Skip if already classified
            if record.get('compilation_summary'):
                continue

            # Get compilation result
            compilation_result = record.get('compilation_result', {})
            if not compilation_result:
                continue

            # Skip successful compilations
            if compilation_result.get('pass', False) and compilation_result.get('complete', False):
                compilation_summary = {
                    'error_counts': {},
                    'total_errors': 0,
                    'status': 'success'
                }
                validation_result = self._validate_compilation_summary(compilation_summary)
                compilation_summary['validation'] = validation_result
                record['compilation_summary'] = compilation_summary
                validation_stats['success'] += 1
                validation_stats['total_processed'] += 1
                if validation_result['is_valid']:
                    validation_stats['valid'] += 1
                else:
                    validation_stats['invalid'] += 1
                continue

            # Get error messages
            errors = compilation_result.get('errors', [])
            if not errors:
                compilation_summary = {
                    'error_counts': {},
                    'total_errors': 0,
                    'status': 'no_errors_found'
                }
                validation_result = self._validate_compilation_summary(compilation_summary)
                compilation_summary['validation'] = validation_result
                record['compilation_summary'] = compilation_summary
                validation_stats['no_errors_found'] += 1
                validation_stats['total_processed'] += 1
                if validation_result['is_valid']:
                    validation_stats['valid'] += 1
                else:
                    validation_stats['invalid'] += 1
                continue

            # Extract error text
            error_texts = []
            for err in errors:
                if isinstance(err, dict):
                    error_texts.append(err.get('data', ''))
                elif isinstance(err, str):
                    error_texts.append(err)

            # Classify errors
            error_counts = {}
            for pattern_name, pattern in ERROR_PATTERNS.items():
                count = 0
                for error_text in error_texts:
                    if re.search(pattern, error_text, re.IGNORECASE):
                        count += 1
                if count > 0:
                    error_counts[pattern_name] = count

            compilation_summary = {
                'error_counts': error_counts,
                'total_errors': len(error_texts)
            }
            validation_result = self._validate_compilation_summary(compilation_summary)
            compilation_summary['validation'] = validation_result
            record['compilation_summary'] = compilation_summary
            classified_count += 1
            validation_stats['has_errors'] += 1
            validation_stats['total_processed'] += 1
            if validation_result['is_valid']:
                validation_stats['valid'] += 1
            else:
                validation_stats['invalid'] += 1
                validation_stats['missing_fields'].extend(validation_result['missing_fields'])

        # Log validation statistics
        self._log_validation_stats(validation_stats, 'compilation')
        if classified_count > 0:
            logger.info(f"Classified compilation errors for {classified_count} records with validation")

        return data_list

    def _clear_attempt_summaries_file(self):
        """Clear the attempt_summaries.json file at the start of reasoning summarization."""
        try:
            os.makedirs(self.attempt_summary_dir, exist_ok=True)
            # Create or truncate the file
            with open(self.attempt_summaries_file, 'w') as f:
                f.write('')
            logger.info(f"Cleared attempt summaries file: {self.attempt_summaries_file}")
        except Exception as e:
            logger.warning(f"Failed to clear attempt summaries file: {e}")

    def _clear_failed_summaries_file(self):
        """Clear the failed_summaries.json file at the start of reasoning summarization."""
        try:
            os.makedirs(self.attempt_summary_dir, exist_ok=True)
            # Create or truncate the file
            with open(self.failed_summaries_file, 'w') as f:
                f.write('')
            logger.info(f"Cleared failed summaries file: {self.failed_summaries_file}")
        except Exception as e:
            logger.warning(f"Failed to clear failed summaries file: {e}")

    def _append_attempt_summary(self, summary_data):
        """Append an attempt summary to attempt_summaries.json (one JSON object per line)."""
        try:
            os.makedirs(self.attempt_summary_dir, exist_ok=True)
            with open(self.attempt_summaries_file, 'a') as f:
                json.dump(summary_data, f)
                f.write('\n')
        except Exception as e:
            logger.warning(f"Failed to append attempt summary: {e}")

    def _append_failed_summary(self, summary_data):
        """Append a failed summary to failed_summaries.json for debugging (one JSON object per line)."""
        try:
            os.makedirs(self.attempt_summary_dir, exist_ok=True)
            with open(self.failed_summaries_file, 'a') as f:
                json.dump(summary_data, f)
                f.write('\n')
        except Exception as e:
            logger.warning(f"Failed to append failed summary: {e}")

    def _load_template(self, template_path):
        if not template_path or not os.path.exists(template_path):
            raise ValueError(f"Template path {template_path} does not exist")

        with open(template_path, "r") as f:
            template = f.read()

        return template

    def _prepare_prompts(self, records):
        """Prepare prompts for LLM summarization."""
        prompts = []
        for record in records:
            # Get formal statement from lean4_code or full_code
            formal_statement = record.get('lean4_code') or record.get('full_code', '')
            reasoning_trace = record.get('model_output', '')

            prompt = self.template.format(
                formal_statement=formal_statement,
                reasoning_trace=reasoning_trace
            )

            prompts.append([
                {
                    "role": "user",
                    "content": prompt
                }
            ])

        return prompts

    def _validate_summary_fields(self, summary, record_idx):
        """
        Validate that the parsed summary contains required fields.

        Args:
            summary: Parsed summary dictionary
            record_idx: Index of the record (for logging)

        Returns:
            dict: Validation result with is_valid, missing_fields, and parse error status
        """
        required_fields = ['summary', 'confidence', 'correctness']
        missing_fields = []
        has_parse_error = 'parse_error' in summary

        for field in required_fields:
            if field not in summary:
                missing_fields.append(field)
            elif summary[field] is None and field != 'lemmas':
                # None is technically present but semantically invalid (except for lemmas which can be empty)
                if field == 'confidence' or field == 'correctness':
                    missing_fields.append(f"{field} (null value)")

        is_valid = len(missing_fields) == 0 and not has_parse_error

        if not is_valid:
            log_msg = f"Record {record_idx}: Invalid summary - "
            if missing_fields:
                log_msg += f"missing/null fields: {missing_fields}"
            if has_parse_error:
                log_msg += f" | parse error: {summary.get('parse_error', 'unknown')}"
            logger.warning(log_msg)

        return {
            'is_valid': is_valid,
            'missing_fields': missing_fields,
            'has_parse_error': has_parse_error
        }

    def _validate_compilation_summary(self, compilation_summary):
        """
        Validate that the compilation summary contains required fields.

        Args:
            compilation_summary: Compiled summary dictionary

        Returns:
            dict: Validation result with is_valid and missing_fields
        """
        required_fields = ['error_counts', 'total_errors']
        missing_fields = []

        for field in required_fields:
            if field not in compilation_summary:
                missing_fields.append(field)

        is_valid = len(missing_fields) == 0

        if not is_valid:
            logger.warning(f"Invalid compilation summary - missing fields: {missing_fields}")

        return {
            'is_valid': is_valid,
            'missing_fields': missing_fields
        }

    def _log_validation_stats(self, stats, summary_type):
        """
        Log validation statistics for a summarization run.

        Args:
            stats: Dictionary containing validation statistics
            summary_type: 'reasoning_trace' or 'compilation'
        """
        total = stats['total_processed']
        if total == 0:
            return

        valid = stats['valid']
        invalid = stats['invalid']
        valid_pct = (valid / total * 100) if total > 0 else 0

        logger.info(f"{summary_type.upper()} Summary Validation Results:")
        logger.info(f"  Total processed: {total}")
        logger.info(f"  Valid: {valid} ({valid_pct:.1f}%)")
        logger.info(f"  Invalid: {invalid} ({100 - valid_pct:.1f}%)")

        if summary_type == 'reasoning_trace':
            parse_errors = stats.get('parse_errors', 0)
            logger.info(f"  Parse errors: {parse_errors}")
            if stats['missing_fields']:
                unique_missing = set(stats['missing_fields'])
                logger.info(f"  Missing fields found: {unique_missing}")

        elif summary_type == 'compilation':
            logger.info(f"  Successful compilations: {stats.get('success', 0)}")
            logger.info(f"  No errors found: {stats.get('no_errors_found', 0)}")
            logger.info(f"  Has errors: {stats.get('has_errors', 0)}")
            if stats['missing_fields']:
                unique_missing = set(stats['missing_fields'])
                logger.info(f"  Missing fields found: {unique_missing}")

    def _parse_summary_response(self, response):
        """Parse JSON summary from LLM response with lenient JSON parsing for LaTeX math."""
        try:
            # Extract reasoning from <think> tags if present
            reasoning = ''
            result_text = response

            # Handle <think> tags more robustly
            if '<think>' in response and '</think>' in response:
                # Extract content between <think> and </think>
                think_match = re.search(r'<think>([\s\S]*?)</think>', response)
                if think_match:
                    reasoning = think_match.group(1).strip()
                    # Get everything after </think>
                    result_text = response.split('</think>', 1)[-1]
            elif '</think>' in response:
                # Malformed: has </think> but no opening <think>
                # Try to extract any text before </think> as reasoning
                parts = response.split('</think>')
                potential_reasoning = parts[0].strip()
                # Remove any closing tags that might be there
                if potential_reasoning:
                    reasoning = potential_reasoning
                result_text = parts[-1]

            # Try to extract JSON from markdown code blocks first
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON object
                json_match = re.search(r'\{[\s\S]*\}', result_text)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = result_text.strip()

            # Parse JSON with lenient handling for LaTeX sequences
            summary = self._parse_json_lenient(json_str)

            # If parsing failed, raise an exception to be caught below
            if summary is None:
                raise ValueError("Failed to parse JSON after all retry attempts")

            # Store reasoning if present
            if reasoning:
                summary['reasoning'] = reasoning

            # Validate expected fields and coerce types
            if 'summary' not in summary:
                summary['summary'] = ''

            if 'confidence' not in summary:
                summary['confidence'] = None
            else:
                # Coerce confidence to int if it's a string representation of a number
                confidence = summary['confidence']
                if isinstance(confidence, str):
                    try:
                        summary['confidence'] = int(confidence)
                    except (ValueError, TypeError):
                        summary['confidence'] = None

            if 'correctness' not in summary:
                summary['correctness'] = None
            else:
                # Coerce correctness to int (1-10 scale) if it's a string representation
                correctness = summary['correctness']
                if isinstance(correctness, str):
                    try:
                        summary['correctness'] = int(correctness)
                    except (ValueError, TypeError):
                        summary['correctness'] = None
                elif isinstance(correctness, bool):
                    # Legacy: convert bool to int (True=10, False=1)
                    summary['correctness'] = 10 if correctness else 1

            if 'lemmas' not in summary:
                summary['lemmas'] = []

            return summary

        except Exception as e:
            logger.warning(f"Failed to parse summary JSON: {e}")
            return {
                'summary': '',
                'confidence': None,
                'correctness': None,
                'lemmas': [],
                'parse_error': str(e),
                'raw_response': response
            }

    def _parse_json_lenient(self, json_str):
        """
        Parse JSON with lenient handling for LaTeX math sequences.
        Handles malformed JSON with unescaped backslashes in string values.

        Args:
            json_str: JSON string potentially containing LaTeX sequences like \\( and \\)

        Returns:
            Parsed dictionary
        """
        try:
            # Try standard JSON parsing first
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            if 'escape' in str(e).lower():
                # Issue is with invalid escape sequences (common with LaTeX)
                # Try using a non-strict decoder
                try:
                    decoder = json.JSONDecoder(strict=False)
                    return decoder.decode(json_str)
                except:
                    pass

                # Last resort: try to work around escape sequence issues
                # Replace problematic escape sequences with safe placeholders
                try:
                    fixed_str = json_str
                    # Replace invalid escape sequences with temporary placeholders
                    replacements = {
                        '\\(': '__ESCAPED_LPAREN__',
                        '\\)': '__ESCAPED_RPAREN__',
                        '\\[': '__ESCAPED_LBRACKET__',
                        '\\]': '__ESCAPED_RBRACKET__',
                        '\\{': '__ESCAPED_LBRACE__',
                        '\\}': '__ESCAPED_RBRACE__',
                        '\\neq': '__ESCAPED_NEQ__',
                        '\\uparrow': '__ESCAPED_UPARROW__',
                    }

                    for original, placeholder in replacements.items():
                        fixed_str = fixed_str.replace(original, placeholder)

                    result = json.loads(fixed_str)

                    # Now fix the strings back
                    def fix_backslashes(obj):
                        if isinstance(obj, str):
                            for original, placeholder in replacements.items():
                                obj = obj.replace(placeholder, original)
                            return obj
                        elif isinstance(obj, dict):
                            return {k: fix_backslashes(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [fix_backslashes(item) for item in obj]
                        return obj

                    return fix_backslashes(result)
                except json.JSONDecodeError:
                    # If placeholder replacement didn't work, try a regex-based approach
                    # to extract fields manually
                    try:
                        result = {}

                        # Try to extract "summary" field
                        summary_match = re.search(r'"summary"\s*:\s*"([^"]*(?:\\.[^"]*)*)"', json_str)
                        if summary_match:
                            # Unescape the string - handle both unicode escapes and simple string content
                            try:
                                result['summary'] = summary_match.group(1).encode().decode('unicode_escape')
                            except:
                                # If unicode_escape fails, just use the raw matched content
                                result['summary'] = summary_match.group(1)

                        # Try to extract "confidence" field (can be a number or quoted string)
                        conf_match = re.search(r'"confidence"\s*:\s*"?([0-9]+)"?', json_str)
                        if conf_match:
                            val = conf_match.group(1)
                            result['confidence'] = int(val) if val else None

                        # Try to extract "correctness" field (can be an integer, boolean, or quoted string)
                        correct_match = re.search(r'"correctness"\s*:\s*"?([0-9]+|true|false|null)"?', json_str)
                        if correct_match:
                            val = correct_match.group(1).lower()
                            if val.isdigit():
                                result['correctness'] = int(val)
                            elif val == 'true':
                                result['correctness'] = 10  # Legacy: true -> 10
                            elif val == 'false':
                                result['correctness'] = 1   # Legacy: false -> 1
                            else:
                                result['correctness'] = None

                        # Try to extract lemmas array
                        lemmas_match = re.search(r'"lemmas"\s*:\s*(\[[\s\S]*?\])', json_str)
                        if lemmas_match:
                            # Try to parse the lemmas array
                            try:
                                lemmas_str = lemmas_match.group(1)
                                # Apply the same placeholder replacement to lemmas
                                for original, placeholder in replacements.items():
                                    lemmas_str = lemmas_str.replace(original, placeholder)
                                result['lemmas'] = json.loads(lemmas_str)
                            except:
                                result['lemmas'] = []

                        if result:
                            return result
                        raise e
                    except:
                        raise e

            raise e
