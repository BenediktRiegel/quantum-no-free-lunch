import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from typing import List
import numpy as np
import json
from data import *
from metrics import *
from numpy.polynomial.polynomial import polyfit

results = np.random.normal(0,1, size= (5,17,25))
results = list(results)
print(results)

x_qbits = 4
num_datapoints = list(range(0, 2**x_qbits + 1))
r_list = list(range(x_qbits + 1))


tableau_palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray',
                   'tab:pink', 'tab:olive', 'tab:cyan']
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s",
            "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"][2:]
for r_idx in range(len(r_list)):
    r = r_list[r_idx]
    rank = 2 ** r
    result_std = [el.std() for el in results[r]]
    plt.scatter(num_datapoints, result_std, label=f"r={rank}", marker=markers[r_idx], c=tableau_palette[r_idx])

    # do linear regression
    b, m = polyfit(num_datapoints, result_std, 1)
    num_datapoints_float = np.arange(2**x_qbits + 1)
    print(num_datapoints_float)
    plt.plot(num_datapoints, b + m * num_datapoints_float, '-', c=tableau_palette[r_idx])

plt.xlabel('No. of Datapoints')
plt.ylabel('Fluctuation in Risk')
plt.legend()
plt.title(f'Fluctuation in Risk for {x_qbits} Qubit Unitary')
plt.tight_layout()
plt.show()
#plt.savefig(f'./plots/{x_qbits}_qubit_exp_fluct.png')
plt.cla()