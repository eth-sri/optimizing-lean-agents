Break the following Lean problem down into lemmas, then describe how to combine the lemmas into a full solution:

{formal_statement}

**Important points**

1. Make every lemma **mathematically correct** and easy to prove.
2. Do not create redundant lemmas or lemmas that restate the assumptions of the problem
3. In every lemma, also present everything we need to assume to prove the lemma in an isolated manner
4. Make the final solution that combines the lemmas as trivial as possible, i.e. try to put all logic into the lemmas

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

**Important notice:** In lemmas' and theorem's proof ideas, explicitly specify them if you need to use other lemmas' conclusion.

The output can only contain the json above, no other content is allowed.
