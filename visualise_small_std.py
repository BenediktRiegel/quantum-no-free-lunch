import numpy as np
import matplotlib.pyplot as plt
from logger import read_logs_regex
from visualisation import calc_lower_bound


def get_small_std_results(all_losses=False):
    attributes = dict(
        schmidt_rank='*',
        std='*',
        losses='*',
        risk='*',
    )
    file_reg = r'./experimental_results/small_std_results/4_points/result_*.txt'
    results = read_logs_regex(file_reg, attributes)
    results_by_std = {}
    num_removed_results = {}
    for res in results:
        risk = res['risk']
        std = res['std']
        losses = res['losses']
        if losses[-1] <= 1e-12 or all_losses:
            if std not in results_by_std.keys():
                results_by_std[std] = []
            results_by_std[std].append(risk)
        else:
            if std not in num_removed_results.keys():
                num_removed_results[std] = 0
            num_removed_results[std] += 1
    print(num_removed_results)
    return results_by_std


def plot_small_std_results(all_losses=False):
    results = get_small_std_results(all_losses=all_losses)
    x = list(results.keys())
    x.sort()
    y = [np.array(results[el]).mean() for el in x]
    plt.scatter(x, y, label='risk')
    for r in [4]:
        lower_bound = calc_lower_bound(r, 4, 2**4)
        y = [lower_bound]*len(x)
        plt.plot(x, y, marker='.', c='tab:gray', linestyle='dashed', label=f"r={r} bound (4 points)")
    plt.legend()
    plt.xlabel('standard deviation')
    plt.ylabel('risk')
    plt.title('Risk depending on the STD (Schmidt Rank 4)')
    plt.tight_layout()
    plt.savefig('./plots/small_std/4_qubits_4_schmidt_rank_exp_4_points.png')


def main():
    plot_small_std_results(all_losses=True)


if __name__ == '__main__':
    main()
