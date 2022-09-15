from winreg import REG_RESOURCE_REQUIREMENTS_LIST
import numpy as np
import matplotlib.pyplot as plt
from logger import read_logs_regex
from visualisation import *

def compress_results():
    attributes = dict(
        schmidt_rank='*',
        num_points='*',
        losses='*',
        risk='*',
    )
    
          
    file_reg = r'./experimental_results/test/merged_results/result_*.txt'
    results = read_logs_regex(file_reg, attributes)
    risks = np.zeros((5, 16))
    res_counter = np.zeros((5, 16))
    for res in results:
        risks[int(np.log2(res['schmidt_rank']))][res['num_points'] - 1] += res['risk']
        res_counter[int(np.log2(res['schmidt_rank']))][res['num_points'] - 1] += 1

    risks = np.array([[risks[r][n] / res_counter[r][n] for n in range(16)] for r in range(5)])
    np.save("./experimental_results/test/risks", risks)
    error_std = np.zeros((5, 16))
    for res in results:
        mean = risks[int(np.log2(res['schmidt_rank']))][res['num_points'] - 1]
        error_std[int(np.log2(res['schmidt_rank']))][res['num_points'] - 1] += (res['risk'] - mean) ** 2
    #error_std = np.array([[1/(10*100) * np.sqrt(x) for x in row] for row in error_std])
    error_std = np.array([[np.sqrt(error_std[r][n]) / res_counter[r][n] for n in range(16)] for r in range(5)])
    
    np.save("./experimental_results/test/error_std", error_std)
        

def plot_results():
    results = np.load("./experimental_results/test/risks.npy")
    err_std = np.load("./experimental_results/test/error_std.npy")

    num_qbits = 4
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:pink', 'tab:olive', 'tab:cyan']

    markers = generate_markers()
    ranks, num_train_pairs = generate_pairs(num_qbits)
    x = [0]
    x.extend(num_train_pairs)
    for rank, color in zip(ranks, color):
        y = [0.94]
        y.extend(results[rank])
        plt.scatter(x, y, color=color, marker=markers[rank], label='r=' + str(2 ** rank))
        plt.errorbar(num_train_pairs, results[rank], err_std[rank], color=color, linestyle='none')
        quantum_bound = [calc_lower_bound(2**rank, num_points, num_qbits ** 2) for num_points in num_train_pairs]
        y = [0.94]
        y.extend(quantum_bound)
        plt.plot(x, y, color=color, linestyle='dashed')
    plt.xlabel('Number of Training Pairs')
    plt.ylabel('Average Risk')
    plt.xticks(range(0, 17))
    plt.ylim(top=1)
    plt.legend()
    plt.title(f'Average Risk for {num_qbits} Qubit Unitary')
    plt.tight_layout()
    plt.savefig(f'./plots/{num_qbits}_qubit_exp_big.png', dpi=600)
    plt.cla()
    print('done')

def main():
    #compress_results()
    plot_results()

if __name__ == '__main__':
    main()