Here is a mathematical problem:

**Problem Statement:** {original_statement}

An LLM has been prompted to formalize the following lemma which is supposed to be a substep of the theorem:

**Statement:** {statement}

**Assumptions:** {assumption}

**Proof Idea:** {proof}

Here are four separate formalizations of the lemma:

```
{lemma_code}
```

Choose the best matching formalization to the lemma according to the statement and the assumptions.

Output ONLY the index number (zero-based) of the best formalization, nothing else. Do not include any explanation after the number.

For example, if formalization 2 is best, output only:
```
2
```
