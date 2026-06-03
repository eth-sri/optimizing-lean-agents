import numpy as np
import pandas as pd

large_pass_file = "data/putnam_solved_gp_48pass.csv"
large_pass = 48

samp_pass = [1, 2, 4, 8, 16, 32, 48]
N = 8

data = pd.read_csv(large_pass_file, sep='\t')

solved_num = data['sum'].values
pass_prob = solved_num / large_pass

print("original solved:", (pass_prob > 0).sum())

# print(pass_prob)

samp_result = np.zeros((len(samp_pass), N))

for i, n in enumerate(samp_pass):
    # print(n)
    this_result = np.zeros((len(pass_prob), N))
    for j, prob in enumerate(pass_prob):
        samples = np.random.binomial(n=n, p=prob, size=N)
        this_result[j] = samples
    this_result = (this_result > 0).astype(int)
    # print(this_result)

    samp_result[i] = this_result.sum(axis=0)

print(samp_result)
print(samp_result.mean(axis=1))
