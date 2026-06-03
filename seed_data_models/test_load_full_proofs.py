"""
Test loading full proof attempts into Problem objects.
"""

from seed_data_models.problem import Problem, load_full_proof_attempts_for_problems


def test_single_problem_load():
    """Test loading full proofs for a single problem."""
    print("=" * 70)
    print("Test: Load Full Proofs for Single Problem")
    print("=" * 70)

    # Create a problem
    problem = Problem(
        origin_problem_id="mathd_algebra_478"
    )

    print(f"\nProblem: {problem.origin_problem_id}")
    print(f"Full proof attempts before load: {problem.full_proof_attempts}")

    # Load full proof attempts
    print("\nLoading full proof attempts...")
    problem.load_full_proof_attempts()

    print(f"\nFull proof attempts after load:")
    print(f"  8b attempts: {len(problem.full_proof_attempts['8b'])}")
    print(f"  32b attempts: {len(problem.full_proof_attempts['32b'])}")

    # Show first attempt from each
    if problem.full_proof_attempts['8b']:
        print(f"\nFirst 8b attempt:")
        attempt = problem.full_proof_attempts['8b'][0]
        print(f"  attempt_id: {attempt.get('attempt_id')}")
        print(f"  pass: {attempt.get('pass')}")
        print(f"  complete: {attempt.get('complete')}")
        print(f"  output_tokens: {attempt.get('detailed_cost', {}).get('output_tokens')}")

    if problem.full_proof_attempts['32b']:
        print(f"\nFirst 32b attempt:")
        attempt = problem.full_proof_attempts['32b'][0]
        print(f"  attempt_id: {attempt.get('attempt_id')}")
        print(f"  pass: {attempt.get('pass')}")
        print(f"  complete: {attempt.get('complete')}")
        print(f"  output_tokens: {attempt.get('detailed_cost', {}).get('output_tokens')}")


def test_multiple_problems_load():
    """Test loading full proofs for multiple problems."""
    print("\n\n" + "=" * 70)
    print("Test: Load Full Proofs for Multiple Problems")
    print("=" * 70)

    # Create multiple problems
    problems = {
        "mathd_algebra_478": Problem(origin_problem_id="mathd_algebra_478"),
        "mathd_algebra_109": Problem(origin_problem_id="mathd_algebra_109"),
        "aime_1983_p1": Problem(origin_problem_id="aime_1983_p1"),
    }

    print(f"\nProblems: {list(problems.keys())}")

    # Load full proof attempts for all
    print("\nLoading full proof attempts for all problems...")
    load_full_proof_attempts_for_problems(problems)

    print(f"\nResults:")
    for problem_id, problem in problems.items():
        num_8b = len(problem.full_proof_attempts.get("8b", []))
        num_32b = len(problem.full_proof_attempts.get("32b", []))
        print(f"  {problem_id:25s}: {num_8b:4d} 8b, {num_32b:4d} 32b")


def test_success_rates():
    """Test calculating success rates from loaded attempts."""
    print("\n\n" + "=" * 70)
    print("Test: Calculate Success Rates")
    print("=" * 70)

    # Create problem and load attempts
    problem = Problem(origin_problem_id="mathd_algebra_478")
    problem.load_full_proof_attempts()

    print(f"\nProblem: {problem.origin_problem_id}")

    # Calculate success rates
    for model_type in ["8b", "32b"]:
        attempts = problem.full_proof_attempts.get(model_type, [])
        if not attempts:
            continue

        total = len(attempts)
        successful = sum(1 for a in attempts if a.get("pass") and a.get("complete"))

        print(f"\n{model_type.upper()} attempts:")
        print(f"  Total: {total}")
        print(f"  Successful: {successful}")
        print(f"  Success rate: {successful/total*100:.1f}%")


if __name__ == "__main__":
    test_single_problem_load()
    test_multiple_problems_load()
    test_success_rates()
