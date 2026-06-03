This is the problem statement we are trying to solve in Lean:

{formal_statement}

The following is a reasoning trace of a prover model attempting to solve the problem:

{reasoning_trace}

Please summarize the reasoning trace, quantify the confidence of the model, and esimate how correct the reasoning trace estimates the proof to be. Please output the summary in the following format:

```json
{{
    "summary": "less than 200 words summarizing the reasoning trace",
    "confidence": 7,
    "correctness": 8
}}
```

**Field Descriptions:**

- `summary` (string): A concise summary (less than 200 words) of the reasoning trace
- `confidence` (integer 0-10): How confident is the model in its reasoning? (0 = very confused/uncertain, 10 = very confident/certain)
- `correctness` (integer 1-10): How correct does the model think its generated proof is? (1 = completely wrong, 10 = completely correct)

**Important:** The output must be valid JSON with exactly these three fields. `confidence` and `correctness` must be integer values, not strings or descriptions.
