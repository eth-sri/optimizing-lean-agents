Break the following math problem into self-contained lemmas as subgoals of solving it.

<informal_prefix>

Instruction:
1. Do not solve the lemmas, but just list the statement of them. 
2. Try to be as concise as possible but do not missing necessary assumptions. 
3. Do not have any dependency between **lemmas** and **original problem**. If you need the conclusion of definition, assumptions from previous lemmas or the originla problem, include them in the currect lemma as well. **Each lemma should be provable without any other addtional information, including the original problem**.
3. The output should be a valid json code block:

```json
{
       "lemma_1": "content of lemma 1",
       "lemma_2": "content of lemma 1",
       "lemma_n": "connent of lemma n"
}
```

Here is an example of the lemmas breakdown:

Problem:

Let a, b, and n be positive integers such that the greatest common divisor of a and b is 1. If x and y are integers satisfying the equation a * x + b * y = a^n + b^n, then there exists an integer k such that x is equal to a^(n-1) + k * b and y is equal to b^(n-1) - k * a.

Output:

```json
{
"lemma_1": "Let a, b, n be positive integers. Define integers x_0 = a^{n-1} and y_0 = b^{n-1}. Then a x_0 + b y_0 = a^n + b^n.",
"lemma_2": "Let a and b be positive integers with gcd(a, b) = 1. Let A, B, x, y be integers such that a x + b y = a A + b B. Then there exists an integer k such that x = A + k b and y = B - k a."
}
```
