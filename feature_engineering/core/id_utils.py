import re


def extract_attempt_base(problem_id: str) -> str:
    """Extract base attempt ID by stripping correction suffixes.
    e.g. 'putnam_1962_a5_r0_p48_corr1_p0_corr2_p0' -> 'putnam_1962_a5_r0_p48'
    """
    return re.sub(r"(_corr\d+_p\d+)+$", "", problem_id)


def extract_origin_problem(problem_id: str) -> str:
    """Extract origin problem from problem_id.
    e.g. 'putnam_1962_a5_r0_p48' -> 'putnam_1962_a5'
    """
    base = extract_attempt_base(problem_id)
    return re.sub(r"_r\d+_p\d+$", "", base)
