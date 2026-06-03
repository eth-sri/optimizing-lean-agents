You are an expert mathematician and Lean programmer. Your task is to break down a previously failed lemma into simpler, more manageable sub-lemmas. You will be given the informal statement of the failed lemma and its formal Lean code.

**Failed Lemma (Informal):**
{statement}

**Failed Lemma (Formal Lean Code):**
```lean
{formal_statement}
```

Now, break the failed lemma down into a new set of simpler lemmas. Follow these important points:

1.  Make every new lemma **mathematically correct** and easier to prove than the failed one.
2.  Do not create redundant lemmas or lemmas that restate existing assumptions.
3.  For every new lemma, state any assumptions required to prove it in an isolated manner.
4.  Make the final solution that combines the new lemmas to prove the original failed lemma as trivial as possible. Put all complex logic into the new lemmas.

The concrete breakdown format must be:

```
problem statement: <restate the FAILED lemma you are trying to solve>

- lemma 1: <new lemma statement, assumptions, and idea of proof>
...
- lemma n: <new lemma statement, assumptions, and idea of proof>

- theorem: <original failed lemma statement, and how to combine the new lemmas to prove it>
```

The output can only contain the breakdown in the above format. Do not include any other content.
