"""
Session data model.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List

from .problem import Problem
from .breakdown_models import Breakdown


@dataclass
class Session:
    """Root container for all loaded data from a run directory."""
    run_dir: Path
    problems: Dict[str, Problem] = field(default_factory=dict)

    def get_problem(self, origin_problem_id: str) -> Optional[Problem]:
        """Get a problem by its origin ID."""
        return self.problems.get(origin_problem_id)

    def get_all_solved_problems(self) -> List[Problem]:
        """Returns list of all problems with at least one solved breakdown."""
        return [p for p in self.problems.values() if p.is_solved()]

    def get_problem_count(self) -> int:
        """Total number of problems in the session."""
        return len(self.problems)

    def get_solved_count(self) -> int:
        """Number of fully solved problems."""
        return len(self.get_all_solved_problems())

    def to_problem_list(self) -> List[Problem]:
        """
        Convert to a list of problems (for compatibility with old render functions).
        Returns list of Problem objects in dict values order.
        """
        return list(self.problems.values())

    def get_total_cost(self, cost_type="cost", exclude_prover_calls: bool = False) -> float:
        """Returns the total cost across all problems.

        Args:
            cost_type: Type of cost to return (e.g., 'cost', 'output_sflops', 'input_tokens')
            exclude_prover_calls: If True, excludes proof attempt costs (useful for seeing just breakdown/formalization costs)
        """
        return sum(problem.get_total_cost(cost_type, exclude_prover_calls=exclude_prover_calls) for problem in self.problems.values())

    def dump(self, output_dir: Path, minified: bool = False):
        """
        Dump the session data to disk.

        Args:
            output_dir: Directory to save data
            minified: If True, saves in flat minified format (no full code/reasoning).
                     If False, saves in hierarchical format organized by problem/breakdown.

        Format when minified=True (flat structure):
            round0/
              problems.json (all problems)
              breakdowns.json (all breakdowns)
              lemmas.json (all lemmas)
              theorems.json (all theorems)
              formalizations.json (all formalizations, no full code)
              proof_attempts.json (all attempts, no full code)

        Format when minified=False (hierarchical structure):
            round0/
              {problem_id}/
                b{breakdown_id}/
                  breakdown.json (individual breakdown)
                  parsed_breakdown.json (parsed data)
                  formalizations.json (all formalizations with full code)
                  lemmas.json (lemma metadata)
                  theorems.json (theorem metadata)
                  proof_attempts/
                    theorem.json (all theorem attempts with full code)
                    0.json, 1.json, ... (per-lemma attempts with full code)
        """
        if minified:
            self.dump_minified(output_dir)
        else:
            self.dump_hierarchical(output_dir)

    def dump_hierarchical(self, output_dir: Path):
        """
        Dump the session data in hierarchical format organized by problem and breakdown.

        Creates:
        - round{N}/
          - {problem_id}/
            - b{breakdown_id}/
              - breakdown.json: Individual breakdown data
              - parsed_breakdown.json: Parsed lemmas and theorem
              - formalizations.json: All formalizations (WITH full code)
              - lemmas.json: Lemma metadata
              - theorems.json: Theorem metadata
              - proof_attempts/
                - theorem.json: All theorem proof attempts (WITH full code)
                - 0.json, 1.json, ...: Per-lemma proof attempts (WITH full code)

        Note: Full code and reasoning traces are included. This format is larger but
        easier to navigate and analyze specific problems/breakdowns.

        Args:
            output_dir: Directory to save hierarchical data
        """
        import json
        from collections import defaultdict

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Dumping session data in hierarchical format to {output_dir}...")

        # Helper function to create breakdown directory structure
        def create_breakdown_structure(breakdown):
            """Create hierarchical structure for a single breakdown."""
            round_id = breakdown.round_id
            breakdown_id = breakdown.breakdown_id
            origin_problem_id = breakdown.origin_problem_id

            # Create breakdown directory
            breakdown_dir = output_dir / f"round{round_id}" / origin_problem_id / f"b{breakdown_id}"
            breakdown_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save breakdown.json
            breakdown_file = breakdown_dir / "breakdown.json"
            with open(breakdown_file, "w") as f:
                json.dump(breakdown.to_dict(), f, indent=2)

            # 2. Save parsed_breakdown.json (if exists)
            if breakdown.parsed_breakdown:
                pb = breakdown.parsed_breakdown
                parsed_file = breakdown_dir / "parsed_breakdown.json"

                # Create a dict with both theorem and lemmas
                parsed_data = {
                    "origin_problem_id": origin_problem_id,
                    "round_id": round_id,
                    "breakdown_id": breakdown_id,
                    "theorem": pb.theorem.to_dict(origin_problem_id, round_id, breakdown_id),
                    "lemmas": [
                        lemma.to_dict(origin_problem_id, round_id, breakdown_id)
                        for lemma in pb.lemmas.values()
                    ]
                }
                with open(parsed_file, "w") as f:
                    json.dump(parsed_data, f, indent=2)

                # 3. Collect and save formalizations
                all_formalizations = []

                # Theorem formalizations
                for form in pb.theorem.formalizations:
                    all_formalizations.append(form.to_dict(
                        origin_problem_id, round_id, breakdown_id, -1,
                        minified=False  # Include full code
                    ))

                # Lemma formalizations
                for lemma in pb.lemmas.values():
                    for form in lemma.formalizations:
                        all_formalizations.append(form.to_dict(
                            origin_problem_id, round_id, breakdown_id, lemma.lemma_id,
                            minified=False  # Include full code
                        ))

                if all_formalizations:
                    formalizations_file = breakdown_dir / "formalizations.json"
                    with open(formalizations_file, "w") as f:
                        json.dump(all_formalizations, f, indent=2)

                # 4. Save lemmas.json and theorems.json
                lemmas_data = [lemma.to_dict(origin_problem_id, round_id, breakdown_id)
                              for lemma in pb.lemmas.values()]
                if lemmas_data:
                    lemmas_file = breakdown_dir / "lemmas.json"
                    with open(lemmas_file, "w") as f:
                        json.dump(lemmas_data, f, indent=2)

                theorems_data = [pb.theorem.to_dict(origin_problem_id, round_id, breakdown_id)]
                theorems_file = breakdown_dir / "theorems.json"
                with open(theorems_file, "w") as f:
                    json.dump(theorems_data, f, indent=2)

                # 5. Save proof_attempts/ directory
                proof_attempts_dir = breakdown_dir / "proof_attempts"
                proof_attempts_dir.mkdir(exist_ok=True)

                # Group proof attempts by lemma_id
                attempts_by_lemma = defaultdict(list)

                # Theorem attempts
                for form in pb.theorem.formalizations:
                    for attempt in form.proof_attempts:
                        attempts_by_lemma[-1].append(attempt.to_dict(
                            formalization_id=form.id,
                            minified=False  # Include full code
                        ))

                # Lemma attempts
                for lemma in pb.lemmas.values():
                    for form in lemma.formalizations:
                        for attempt in form.proof_attempts:
                            attempts_by_lemma[lemma.lemma_id].append(attempt.to_dict(
                                formalization_id=form.id,
                                minified=False  # Include full code
                            ))

                # Save each lemma's attempts to separate files
                for lemma_id, attempts in attempts_by_lemma.items():
                    if lemma_id == -1:
                        filename = "theorem.json"
                    else:
                        filename = f"{lemma_id}.json"

                    attempt_file = proof_attempts_dir / filename
                    with open(attempt_file, "w") as f:
                        json.dump(attempts, f, indent=2)

                return len(all_formalizations), len(attempts_by_lemma)

            return 0, 0

        # Process all problems
        total_breakdowns = 0
        total_formalizations = 0
        total_attempts_files = 0

        for problem in self.problems.values():
            for breakdown in problem.breakdowns.values():
                form_count, attempt_count = create_breakdown_structure(breakdown)
                total_breakdowns += 1
                total_formalizations += form_count
                total_attempts_files += attempt_count

        print(f"\n✅ Hierarchical dump complete!")
        print(f"  - {len(self.problems)} problems")
        print(f"  - {total_breakdowns} breakdowns")
        print(f"  - {total_formalizations} formalizations")
        print(f"  - {total_attempts_files} proof attempt files")
        print(f"\nSaved to: {output_dir}")
        print(f"\nStructure: round{{N}}/{{problem_id}}/b{{breakdown_id}}/...")

    def load_breakdown(self, origin_problem_id: str, round_id: int, breakdown_id: int,
                      hierarchical_dir: Path):
        """
        Progressively load full breakdown data from hierarchical format.

        This enriches the existing breakdown in the session with full data that is NOT
        in the minified format:
        - Full code from formalizations (reasoning, prompts, complete code)
        - Full breakdown text with reasoning traces
        - Complete parsed breakdown data
        - All metadata

        Note: This does NOT load proof attempts. Use load_attempts() for that.

        Typical workflow:
            1. Load session from minified: session = Session.load_from_minified(minified_dir)
            2. Load specific breakdown: session.load_breakdown("problem_id", 0, 0, hierarchical_dir)
            3. Load specific attempts: session.load_attempts("problem_id", 0, 0, -1, hierarchical_dir)

        Args:
            origin_problem_id: Problem ID (e.g., "mathd_algebra_209")
            round_id: Round number (usually 0 for initial, 1+ for recursive)
            breakdown_id: Breakdown ID within the problem (0, 1, 2, ...)
            hierarchical_dir: Root directory of hierarchical dump

        Raises:
            ValueError: If the breakdown doesn't exist in the session or hierarchical dir
        """
        import json

        hierarchical_dir = Path(hierarchical_dir)

        # Find the breakdown in the current session
        breakdown_key = (origin_problem_id, round_id, breakdown_id)
        problem = self.problems.get(origin_problem_id)

        if not problem:
            raise ValueError(f"Problem {origin_problem_id} not found in session. "
                           f"Load session first with Session.load_from_minified()")

        breakdown = problem.breakdowns.get(breakdown_key)
        if not breakdown:
            raise ValueError(f"Breakdown {breakdown_key} not found in problem. "
                           f"Available breakdowns: {list(problem.breakdowns.keys())}")

        # Path to breakdown directory
        breakdown_dir = hierarchical_dir / f"round{round_id}" / origin_problem_id / f"b{breakdown_id}"

        if not breakdown_dir.exists():
            raise ValueError(f"Breakdown directory not found: {breakdown_dir}")

        # 1. Load breakdown.json (full breakdown with reasoning)
        breakdown_file = breakdown_dir / "breakdown.json"
        if breakdown_file.exists():
            with open(breakdown_file) as f:
                breakdown_data = json.load(f)
                # Update breakdown with full data (reasoning traces, prompts, etc.)
                if "informal_breakdown" in breakdown_data:
                    breakdown.informal_breakdown = breakdown_data["informal_breakdown"]
                if "informal_breakdown_reasoning" in breakdown_data:
                    breakdown.informal_breakdown_reasoning = breakdown_data["informal_breakdown_reasoning"]
                if "breakdown_prompt" in breakdown_data:
                    breakdown.breakdown_prompt = breakdown_data["breakdown_prompt"]

        # 2. Load parsed_breakdown.json (full parsed data)
        parsed_file = breakdown_dir / "parsed_breakdown.json"
        if parsed_file.exists():
            with open(parsed_file) as f:
                parsed_data = json.load(f)
                # This already exists in the breakdown, but we may enrich it with additional fields
                # For now, skip since parsed_breakdown was already loaded from minified

        # 3. Load formalizations.json (WITH full code, reasoning, prompts)
        formalizations_file = breakdown_dir / "formalizations.json"
        if formalizations_file.exists():
            with open(formalizations_file) as f:
                formalizations_data = json.load(f)

                # Map formalization_id to formalization data
                # Get formalization_id from metadata.formalization_id
                forms_by_id = {f["metadata"]["formalization_id"]: f for f in formalizations_data}

                # Update each formalization in the parsed_breakdown
                if breakdown.parsed_breakdown:
                    pb = breakdown.parsed_breakdown

                    # Update theorem formalizations
                    for form in pb.theorem.formalizations:
                        if form.id in forms_by_id:
                            form_data = forms_by_id[form.id]
                            # Handle both hierarchical and minified formats
                            form.full_code = form_data.get("full_code", form_data.get("formal_statement", ""))
                            form.reasoning = form_data.get("reasoning", form_data.get("formalization_reasoning", ""))
                            form.prompt = form_data.get("prompt", "")
                            form.prompt_type = form_data.get("prompt_type", "")

                    # Update lemma formalizations
                    for lemma in pb.lemmas.values():
                        for form in lemma.formalizations:
                            if form.id in forms_by_id:
                                form_data = forms_by_id[form.id]
                                # Handle both hierarchical and minified formats
                                form.full_code = form_data.get("full_code", form_data.get("formal_statement", ""))
                                form.reasoning = form_data.get("reasoning", form_data.get("formalization_reasoning", ""))
                                form.prompt = form_data.get("prompt", "")
                                form.prompt_type = form_data.get("prompt_type", "")

        print(f"✅ Loaded breakdown data for {origin_problem_id} round{round_id} b{breakdown_id}")
        print(f"   (Proof attempts not loaded - use load_attempts() to load those)")

    def load_attempts(self, origin_problem_id: str, round_id: int, breakdown_id: int,
                     lemma_id: int, hierarchical_dir: Path):
        """
        Progressively load proof attempts for a specific lemma/theorem.

        This loads all proof attempts with full data:
        - Full Lean code
        - Reasoning traces
        - Compilation results with full error messages
        - Prompts and intermediate results

        Args:
            origin_problem_id: Problem ID (e.g., "mathd_algebra_209")
            round_id: Round number
            breakdown_id: Breakdown ID
            lemma_id: Lemma ID (-1 for theorem, 0+ for lemmas)
            hierarchical_dir: Root directory of hierarchical dump

        Raises:
            ValueError: If the breakdown or attempts don't exist
        """
        import json
        from .proof_attempt import ProofAttempt

        hierarchical_dir = Path(hierarchical_dir)

        # Find the breakdown in the current session
        breakdown_key = (origin_problem_id, round_id, breakdown_id)
        problem = self.problems.get(origin_problem_id)

        if not problem:
            raise ValueError(f"Problem {origin_problem_id} not found in session")

        breakdown = problem.breakdowns.get(breakdown_key)
        if not breakdown or not breakdown.parsed_breakdown:
            raise ValueError(f"Breakdown {breakdown_key} or parsed_breakdown not found")

        # Path to proof attempts directory
        attempts_dir = (hierarchical_dir / f"round{round_id}" / origin_problem_id /
                       f"b{breakdown_id}" / "proof_attempts")

        if not attempts_dir.exists():
            raise ValueError(f"Proof attempts directory not found: {attempts_dir}")

        # Determine filename
        if lemma_id == -1:
            filename = "theorem.json"
            target = breakdown.parsed_breakdown.theorem
        else:
            filename = f"{lemma_id}.json"
            target = breakdown.parsed_breakdown.lemmas.get(lemma_id)
            if not target:
                raise ValueError(f"Lemma {lemma_id} not found in breakdown")

        attempts_file = attempts_dir / filename
        if not attempts_file.exists():
            raise ValueError(f"Attempts file not found: {attempts_file}")

        # Load attempts
        with open(attempts_file) as f:
            attempts_data = json.load(f)

        # Group by formalization_id
        attempts_by_form_id = {}
        for attempt_data in attempts_data:
            form_id = attempt_data["metadata"]["formalization_id"]
            if form_id not in attempts_by_form_id:
                attempts_by_form_id[form_id] = []
            attempts_by_form_id[form_id].append(attempt_data)

        # Update each formalization's proof_attempts
        for form in target.formalizations:
            if form.id in attempts_by_form_id:
                # Clear existing attempts (from minified) and replace with full attempts
                form.proof_attempts = []

                for attempt_data in sorted(attempts_by_form_id[form.id],
                                          key=lambda x: x["metadata"]["attempt_id"]):
                    attempt = ProofAttempt.from_dict(attempt_data)
                    form.proof_attempts.append(attempt)

        lemma_name = "theorem" if lemma_id == -1 else f"lemma {lemma_id}"
        print(f"✅ Loaded {len(attempts_data)} proof attempts for {origin_problem_id} "
              f"round{round_id} b{breakdown_id} {lemma_name}")

    def dump_minified(self, output_dir: Path):
        """
        Dump the session data in minified flat format organized by rounds.

        Creates:
        - round0/
          - problems.json: All problem metadata
          - breakdowns.json: All breakdown metadata
          - lemmas.json: All lemma metadata
          - theorems.json: All theorem metadata
          - formalizations.json: All formalizations (without proof_attempts)
          - proof_attempts.json: All proof attempts (minified - no full code/reasoning)
        - round1/
          ... (same structure for recursive attempts)
        - round2/
          ... (and so on)

        Note: Only minified proof attempts are saved (metadata, summaries, compilation results).
        Full code and reasoning traces are excluded to save space. Access original full_records
        if you need the complete data.

        Args:
            output_dir: Directory to save minified data
        """
        import json
        from collections import defaultdict

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Organize data by round
        data_by_round = defaultdict(lambda: {
            "problems": [],
            "breakdowns": [],
            "lemmas": [],
            "theorems": [],
            "formalizations": [],
            "proof_attempts": [],
            "proof_attempts_with_proof": []
        })

        # Collect problems recursively
        def collect_problem(problem):
            """Recursively collect data from a problem and its recursive attempts."""

            # Add problem
            data_by_round[0]["problems"].append(problem.to_dict())

            # Traverse all breakdowns
            for breakdown in problem.breakdowns.values():
                round_id = breakdown.round_id
                round_data = data_by_round[round_id]

                # Add breakdown
                round_data["breakdowns"].append(breakdown.to_dict(minified=True))

                # If breakdown has parsed_breakdown, traverse lemmas/theorems/formalizations
                if breakdown.parsed_breakdown:
                    pb = breakdown.parsed_breakdown

                    # Add theorem
                    round_data["theorems"].append(pb.theorem.to_dict(
                        breakdown.origin_problem_id,
                        breakdown.round_id,
                        breakdown.breakdown_id
                    ))

                    # Traverse theorem formalizations
                    for form in pb.theorem.formalizations:
                        round_data["formalizations"].append(form.to_dict(
                            breakdown.origin_problem_id,
                            breakdown.round_id,
                            breakdown.breakdown_id,
                            -1,  # theorem
                            minified=True
                        ))

                        # Traverse theorem proof attempts
                        for attempt in form.proof_attempts:
                            round_data["proof_attempts"].append(attempt.to_dict(
                                formalization_id=form.id,
                                minified=True
                            ))
                            round_data["proof_attempts_with_proof"].append(attempt.to_dict(
                                formalization_id=form.id,
                                minified=True,
                                include_code=True
                            ))

                    # Add lemmas
                    for lemma in pb.lemmas.values():
                        round_data["lemmas"].append(lemma.to_dict(
                            breakdown.origin_problem_id,
                            breakdown.round_id,
                            breakdown.breakdown_id
                        ))

                        # Traverse lemma formalizations
                        for form in lemma.formalizations:
                            round_data["formalizations"].append(form.to_dict(
                                breakdown.origin_problem_id,
                                breakdown.round_id,
                                breakdown.breakdown_id,
                                lemma.lemma_id,
                                minified=True
                            ))

                            # Traverse lemma proof attempts
                            for attempt in form.proof_attempts:
                                round_data["proof_attempts"].append(attempt.to_dict(
                                    formalization_id=form.id,
                                    minified=True
                                ))
                                round_data["proof_attempts_with_proof"].append(attempt.to_dict(
                                    formalization_id=form.id,
                                    minified=True,
                                    include_code=True
                                ))

            # Recursively handle recursive_attempts
            for recursive_problem in problem.recursive_attempts:
                collect_problem(recursive_problem)

        # Collect data from all problems
        for problem in self.problems.values():
            collect_problem(problem)

        # Save data for each round
        for round_id in sorted(data_by_round.keys()):
            round_data = data_by_round[round_id]
            round_dir = output_dir / f"round{round_id}"
            round_dir.mkdir(parents=True, exist_ok=True)

            # Save flat files
            with open(round_dir / "problems.json", "w") as f:
                json.dump(round_data["problems"], f, indent=2)

            with open(round_dir / "breakdowns.json", "w") as f:
                json.dump(round_data["breakdowns"], f, indent=2)

            with open(round_dir / "lemmas.json", "w") as f:
                json.dump(round_data["lemmas"], f, indent=2)

            with open(round_dir / "theorems.json", "w") as f:
                json.dump(round_data["theorems"], f, indent=2)

            with open(round_dir / "formalizations.json", "w") as f:
                json.dump(round_data["formalizations"], f, indent=2)

            with open(round_dir / "proof_attempts.json", "w") as f:
                json.dump(round_data["proof_attempts"], f, indent=2)

            with open(round_dir / "proof_attempts_with_proof.jsonl", "w") as f:
                for entry in round_data["proof_attempts_with_proof"]:
                    f.write(json.dumps(entry) + "\n")

            print(f"Round {round_id} saved to {round_dir}")
            print(f"  - {len(round_data['problems'])} problems")
            print(f"  - {len(round_data['breakdowns'])} breakdowns")
            print(f"  - {len(round_data['lemmas'])} lemmas")
            print(f"  - {len(round_data['theorems'])} theorems")
            print(f"  - {len(round_data['formalizations'])} formalizations")
            print(f"  - {len(round_data['proof_attempts'])} proof attempts")

        print(f"\nMinified data saved to {output_dir}")

    @classmethod
    def load_from_minified(cls, minified_dir: Path, verbose: bool = True, load_proof_code: bool = False) -> 'Session':
        """
        Load session data from minified flat format.

        Args:
            minified_dir: Directory containing minified data (with round0/, round1/, etc.)
            verbose: If True, print loading progress
            load_proof_code: If True, load full Lean code from proof_attempts_with_proof.jsonl

        Returns:
            Session object with all data loaded
        """
        import json
        from collections import defaultdict

        minified_dir = Path(minified_dir)
        if not minified_dir.exists():
            raise ValueError(f"Minified directory {minified_dir} does not exist")

        if verbose:
            print(f"Loading minified data from {minified_dir}...")

        # Find all round directories
        round_dirs = sorted([d for d in minified_dir.iterdir() if d.is_dir() and d.name.startswith("round")])
        if not round_dirs:
            raise ValueError(f"No round directories found in {minified_dir}")

        if verbose:
            print(f"Found {len(round_dirs)} rounds: {[d.name for d in round_dirs]}")

        # Data structures to hold loaded data organized by (origin_problem_id, round_id, breakdown_id, lemma_id, formalization_id)
        problems_by_id = {}  # {origin_problem_id: Problem}
        breakdowns_by_key = {}  # {(origin_problem_id, round_id, breakdown_id): Breakdown}
        lemmas_by_key = {}  # {(origin_problem_id, round_id, breakdown_id, lemma_id): Lemma}
        theorems_by_key = {}  # {(origin_problem_id, round_id, breakdown_id): Theorem}
        formalizations_by_key = defaultdict(list)  # {(origin_problem_id, round_id, breakdown_id, lemma_id): [Formalization]}
        proof_attempts_by_key = defaultdict(list)  # {(origin_problem_id, round_id, breakdown_id, lemma_id, formalization_id): [ProofAttempt]}

        # Load data from each round
        for round_dir in round_dirs:
            round_id = int(round_dir.name.replace("round", ""))
            if verbose:
                print(f"\nLoading {round_dir.name}...")

            # Load problems (only from round 0)
            if round_id == 0:
                problems_file = round_dir / "problems.json"
                if problems_file.exists():
                    with open(problems_file) as f:
                        problems_data = json.load(f)
                        for problem_data in problems_data:
                            problem = Problem.from_dict(problem_data)
                            problems_by_id[problem.origin_problem_id] = problem
                        if verbose:
                            print(f"  - Loaded {len(problems_data)} problems")

            # Load breakdowns
            breakdowns_file = round_dir / "breakdowns.json"
            if breakdowns_file.exists():
                with open(breakdowns_file) as f:
                    breakdowns_data = json.load(f)
                    for breakdown_data in breakdowns_data:
                        breakdown = Breakdown.from_dict(breakdown_data)
                        key = (breakdown.origin_problem_id, breakdown.round_id, breakdown.breakdown_id)
                        breakdowns_by_key[key] = breakdown
                    if verbose:
                        print(f"  - Loaded {len(breakdowns_data)} breakdowns")

            # Load lemmas
            lemmas_file = round_dir / "lemmas.json"
            if lemmas_file.exists():
                with open(lemmas_file) as f:
                    lemmas_data = json.load(f)
                    for lemma_data in lemmas_data:
                        from .lemma import Lemma
                        lemma = Lemma.from_dict(lemma_data)
                        metadata = lemma_data["metadata"]
                        key = (metadata["origin_problem_id"], metadata["round_id"], metadata["breakdown_id"], metadata["lemma_id"])
                        lemmas_by_key[key] = lemma
                    if verbose:
                        print(f"  - Loaded {len(lemmas_data)} lemmas")

            # Load theorems
            theorems_file = round_dir / "theorems.json"
            if theorems_file.exists():
                with open(theorems_file) as f:
                    theorems_data = json.load(f)
                    for theorem_data in theorems_data:
                        from .theorem import Theorem
                        theorem = Theorem.from_dict(theorem_data)
                        metadata = theorem_data["metadata"]
                        key = (metadata["origin_problem_id"], metadata["round_id"], metadata["breakdown_id"])
                        theorems_by_key[key] = theorem
                    if verbose:
                        print(f"  - Loaded {len(theorems_data)} theorems")

            # Load formalizations
            formalizations_file = round_dir / "formalizations.json"
            if formalizations_file.exists():
                with open(formalizations_file) as f:
                    formalizations_data = json.load(f)
                    for form_data in formalizations_data:
                        from .formalization import Formalization
                        formalization = Formalization.from_dict(form_data)
                        metadata = form_data["metadata"]
                        key = (metadata["origin_problem_id"], metadata["round_id"], metadata["breakdown_id"], metadata["lemma_id"])
                        formalizations_by_key[key].append(formalization)
                    if verbose:
                        print(f"  - Loaded {len(formalizations_data)} formalizations")

            # Load proof attempts
            proof_attempts_file = round_dir / "proof_attempts.json"
            if proof_attempts_file.exists():
                # Optionally load proof code and errors from enriched JSONL
                code_index = {}
                errors_index = {}
                if load_proof_code:
                    proof_code_file = round_dir / "proof_attempts_with_proof.jsonl"
                    if proof_code_file.exists():
                        with open(proof_code_file) as f:
                            for line in f:
                                rec = json.loads(line)
                                m = rec["metadata"]
                                idx_key = (
                                    m["origin_problem_id"], m["round_id"], m["breakdown_id"],
                                    m["lemma_id"], m["formalization_id"], m["attempt_id"],
                                    m.get("correction_round_id", 0), m.get("initial_attempt_id", 0),
                                )
                                code = rec.get("full_code") or rec.get("code")
                                if code is not None:
                                    code_index[idx_key] = code
                                errors = rec.get("compilation_result", {}).get("errors")
                                if errors is not None:
                                    errors_index[idx_key] = errors
                        if verbose:
                            print(f"  - Loaded {len(code_index)} proof codes and {len(errors_index)} error lists from JSONL")
                    elif verbose:
                        print(f"  - Warning: load_proof_code=True but {proof_code_file} not found")

                with open(proof_attempts_file) as f:
                    proof_attempts_data = json.load(f)
                    for attempt_data in proof_attempts_data:
                        if load_proof_code:
                            m = attempt_data["metadata"]
                            idx_key = (
                                m["origin_problem_id"], m["round_id"], m["breakdown_id"],
                                m["lemma_id"], m["formalization_id"], m["attempt_id"],
                                m.get("correction_round_id", 0), m.get("initial_attempt_id", 0),
                            )
                            code = code_index.get(idx_key)
                            if code is not None:
                                attempt_data["code"] = code
                            errors = errors_index.get(idx_key)
                            if errors is not None:
                                attempt_data["compilation_result"]["errors"] = errors
                        from .proof_attempt import ProofAttempt
                        attempt = ProofAttempt.from_dict(attempt_data)
                        metadata = attempt_data["metadata"]
                        key = (metadata["origin_problem_id"], metadata["round_id"], metadata["breakdown_id"], metadata["lemma_id"], metadata["formalization_id"])
                        proof_attempts_by_key[key].append(attempt)
                    if verbose:
                        print(f"  - Loaded {len(proof_attempts_data)} proof attempts")

        # Link everything together
        if verbose:
            print("\nLinking data structures...")

        # Link proof_attempts -> formalizations
        lemma_attempts_linked = 0
        theorem_attempts_linked = 0
        for key, attempts in proof_attempts_by_key.items():
            origin_problem_id, round_id, breakdown_id, lemma_id, form_id = key
            form_key = (origin_problem_id, round_id, breakdown_id, lemma_id)
            if form_key in formalizations_by_key:
                for formalization in formalizations_by_key[form_key]:
                    if formalization.id == form_id:
                        formalization.proof_attempts = sorted(attempts, key=lambda a: a.attempt_id)
                        if lemma_id == -1:
                            theorem_attempts_linked += len(attempts)
                        else:
                            lemma_attempts_linked += len(attempts)
                        break
        if verbose and (lemma_attempts_linked > 0 or theorem_attempts_linked > 0):
            print(f"  - Linked {lemma_attempts_linked} lemma proof attempts, {theorem_attempts_linked} theorem proof attempts")

        # Link formalizations -> lemmas/theorems
        for key, formalizations in formalizations_by_key.items():
            origin_problem_id, round_id, breakdown_id, lemma_id = key
            sorted_forms = sorted(formalizations, key=lambda f: f.id)

            if lemma_id == -1:
                # Theorem
                theorem_key = (origin_problem_id, round_id, breakdown_id)
                if theorem_key in theorems_by_key:
                    theorems_by_key[theorem_key].formalizations = sorted_forms
            else:
                # Lemma
                lemma_key = (origin_problem_id, round_id, breakdown_id, lemma_id)
                if lemma_key in lemmas_by_key:
                    lemmas_by_key[lemma_key].formalizations = sorted_forms

        # Link lemmas/theorems -> ParsedBreakdown
        for breakdown_key, breakdown in breakdowns_by_key.items():
            origin_problem_id, round_id, breakdown_id = breakdown_key

            # Get theorem
            theorem_key = (origin_problem_id, round_id, breakdown_id)
            if theorem_key not in theorems_by_key:
                continue  # Skip if no theorem (parse failure)

            theorem = theorems_by_key[theorem_key]

            # Get lemmas
            lemmas_dict = {}
            for lemma_key, lemma in lemmas_by_key.items():
                l_origin, l_round, l_breakdown, l_id = lemma_key
                if l_origin == origin_problem_id and l_round == round_id and l_breakdown == breakdown_id:
                    lemmas_dict[l_id] = lemma

            # Create ParsedBreakdown
            from .breakdown_models import ParsedBreakdown
            parsed_breakdown = ParsedBreakdown.from_dict(theorem, lemmas_dict)

            # Set parser cost from saved _parser_detailed_cost if available
            if breakdown._parser_detailed_cost:
                parsed_breakdown.detailed_cost = breakdown._parser_detailed_cost

            breakdown.parsed_breakdown = parsed_breakdown

        # Link breakdowns -> problems
        for breakdown_key, breakdown in breakdowns_by_key.items():
            origin_problem_id, round_id, breakdown_id = breakdown_key

            # Find or create the problem
            if origin_problem_id not in problems_by_id:
                # Create a new problem (for recursive attempts that may not be in problems.json)
                problems_by_id[origin_problem_id] = Problem(origin_problem_id=origin_problem_id)

            problem = problems_by_id[origin_problem_id]

            # Add breakdown to problem
            bd_tuple_key = (origin_problem_id, round_id, breakdown_id)
            problem.breakdowns[bd_tuple_key] = breakdown

        # Handle recursive attempts
        # Recursive problems have origin_problem_id that matches a lemma UID from a parent problem
        # We need to link them as recursive_attempts
        # For now, we'll skip this complex linking and just return the flat structure

        if verbose:
            print(f"\n✅ Loaded {len(problems_by_id)} problems with {len(breakdowns_by_key)} breakdowns")

        # Create session
        session = cls(run_dir=minified_dir, problems=problems_by_id)

        # Calculate initial_attempt_index for correction round tracking
        # This ensures correction attempts are properly linked to their originals
        from seed_data_models.model_loader import DataLoader
        loader = DataLoader(minified_dir)
        # NOTE: Don't recompute initial_attempt_index - use values from JSON
        # loader._calculate_initial_attempt_indices(session)

        return session
