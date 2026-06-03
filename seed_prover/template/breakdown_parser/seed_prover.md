Please convert this informal breakdown of the problem into a structured format:

{informal_prefix}

**Important points**

The concrete breakdown format should be HTML and should have the following structure:

```
<solution>
    {{
        "lemmas": [
            {{
                "id": <lemma number>,
                "statement": <lemma statement>,
                "assumption": <state the necessary assumptions for the lemma>,
                "proof": <idea of the proof of the lemma in natural language>
            }}
        ],
        "theorem": {{
            "statement": <repeat the problem statement>,
            "proof": <idea of the proof, how to combine the lemmas into the final solution>
        }}
    }}
</solution>
```

The output can only contain the breakdown in the above format, any other contents are not allowed. Do not introduce new keys to the JSON in the output. Also remember to keep the solution nested in the <solution></solution> tags. **Important** Make sure that the JSON code you produce compiles, do not use inconsistent quotation marks, or incorrect escape characters for LaTeX text or similar.
