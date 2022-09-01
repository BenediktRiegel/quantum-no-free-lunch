from logger import read_logs_regex
import gc
from visualisation import generate_risk_plot
import numpy as np
import matplotlib.pyplot as plt


def main():
    # file_paths = r'./experimental_results/std_results/result*'
    file_paths = r'./experimental_results/std_results/result_test.txt'
    attributes = dict(
        schmidt_rank="*",
        num_points="*",
        std="*",
        mean="*",
        losses="*",
    )
    results = read_logs_regex(file_paths, attributes)
    print(f"num of results = {len(results)}")
    preped_results = []
    for result in results:
        print(result['losses'][-1])
        if result['losses'][-1] == 0.0:
            preped_results.append(result)
    # del results
    # gc.collect()
    # schmidt_rank, num_points, std, mean, losses, risk, train_time, qnn, unitary
    unique_schmidt = dict()
    print(f"num of preped results {len(preped_results)}")
    for res in preped_results:
        std = np.round(res['std'], 0)
        if res['schmidt_rank'] not in unique_schmidt.keys():
            unique_schmidt[res['schmidt_rank']] = dict()
        if unique_schmidt[res['schmidt_rank']][std] not in unique_schmidt[res['schmidt_rank']].keys():
            unique_schmidt[res['schmidt_rank']][std] = [None]*16
        if unique_schmidt[res['schmidt_rank']][std][res['num_points']+1] is None:
            unique_schmidt[res['schmidt_rank']][std][res['num_points'] + 1] = []
        unique_schmidt[res['schmidt_rank']][std][res['num_points'] + 1].append(res['risk'])
    print(f"schmidt_ranks = {unique_schmidt.keys()}")
    for s_rank in unique_schmidt.keys():
        print(f"\tschmidt_ranks = {unique_schmidt.keys()} contains stds = {unique_schmidt[s_rank]}")

    return
    from visualisation import generate_markers, calc_lower_bound
    num_datapoints = list(range(17))
    r_list = [2**i for i in range(5)]
    x_qbits = 4

    for s_rank in unique_schmidt.keys():
        tableau_palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown',
                           'tab:pink', 'tab:olive', 'tab:cyan']
        markers = generate_markers()[2:]
        for r_idx in range(len(r_list)):
            r = r_list[r_idx]
            rank = 2 ** r
            d = 2 ** x_qbits
            quantum_bound = [calc_lower_bound(rank, num_points, d) for num_points in num_datapoints]
            plt.plot(num_datapoints, quantum_bound, label=f"r={rank} bound", marker='.', c='tab:gray',
                     linestyle='dashed')

        std_idx = 0
        for std in unique_schmidt[s_rank].keys():
            result_mean = [0] + [unique_schmidt[s_rank][std][num_points].mean() for num_points in num_datapoints[1:]]
            plt.scatter(num_datapoints, result_mean, label=f"std={std}", marker=markers[std_idx], c=tableau_palette[std_idx])
            std_idx += 1
        plt.xlabel('No. of Datapoints')
        plt.ylabel('Average Risk')
        plt.legend()
        plt.title(f'Average Risk for {x_qbits} Qubit Unitary and schmidt rank {s_rank}')
        plt.tight_layout()
        plt.savefig(f'./plots/std/{x_qbits}_qubit_{s_rank}_schmidt_rank_exp.png')
        plt.cla()


if __name__ == '__main__':
    main()
