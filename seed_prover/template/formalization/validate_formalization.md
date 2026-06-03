Here is a mathematical problem:

**Problem Statement:** {original_statement}

An LLM has been prompted to formalize the following lemma which is supposed to be a substep of the theorem:

**Statement:** {statement}

**Assumptions:** {assumption}

**Proof Idea:** {proof}

The generated Lean statement is the following:

{compiled_code}

To sanity check this formalization, does this formalization formalize the statement or does the statement seem to formalizing something totally different like the problem statement itself? The formalization might have extra assumptions which might be artifacts from other parts of the problem solving process, this sanity check is just to check if there is any resemblence.

Your answer should have the following format (the answer should be lowercase yes/no):

<verdict>yes/no</verdict>