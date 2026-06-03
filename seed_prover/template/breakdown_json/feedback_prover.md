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

The concrete breakdown format should be a valid json. The output format and what you need to include is the following:

```json
{{
    "lemmas": [
        {{
            "id": <lemma number>,
            "statement": <lemma statement>,
            "assumption": <state the necessary assumptions for the lemma>,
            "proof": <idea of the proof of the lemma in natural language, if you need to use other lemmas, specify it in the proof idea>
        }}
    ],
    "theorem": {{
        "statement": <repeat the problem statement>,
        "proof": <idea of the proof, how to combine the lemmas into the final solution>
    }}
}}
```
