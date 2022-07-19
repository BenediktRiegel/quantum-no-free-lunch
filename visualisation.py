import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from typing import List
import numpy as np
import json


# create plot analogous average risk vs. training pairs


def set_fontsize(text=10, title=10, labels=10, xtick=10, ytick=10, legend=10):
    plt.rc('font', size=text)  # controls default text size
    plt.rc('axes', titlesize=title)  # fontsize of the title
    plt.rc('axes', labelsize=labels)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=xtick)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=ytick)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=legend)  # fontsize of the legend


def generate_markers():
    return [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s",
            "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]


def generate_pairs(max_rank):
    # construct pairs on which to evaluate experiment
    ranks = [i for i in range(0, max_rank + 1)]
    num_train_pairs = [i for i in range(1, 2 ** max_rank + 1)]
    return ranks, num_train_pairs


def get_result(ranks, num_train_pairs):
    data_points = get_test_results(ranks, num_train_pairs)
    return data_points


def plot_fig2():
    results = np.load('./experimental_results/exp1/result.npy')
    for i in range(len(results)):
        plt.plot(range(1, 3), results[i, :], label=f"r={2**i}")
    plt.xlabel('Number of Training pairs t')
    plt.ylabel('Average Risk')
    plt.title('Average Riks vs. Number of Training pairs')
    plt.legend()
    plt.savefig('./experimental_results/exp1/result.png')
    plt.cla()


def plot_fig3():
    with open('./experimental_results/exp2/result.json', 'r') as f:
        result = json.load(f)
        f.close()

    # color_palette = generate_color_palette(num_ranks)
    num_qbits = 6

    color = cm.rainbow(np.linspace(0, 10, num_qbits + 1))
    markers = generate_markers()
    ranks, num_train_pairs = generate_pairs(num_qbits + 1)

    for rank, color in zip(ranks, color):
        plt.plot(num_train_pairs, result[rank], color=color, marker=markers[rank], label='r=' + str(2 ** rank))

    # create deterministic line
    def f(t):
        return (1 - 1 / (2 ** num_qbits)) * (1 - t / (2 ** num_qbits))

    xvals = np.linspace(0, 2 ** num_qbits, 200)
    yvals = list(map(f, xvals))
    plt.plot(xvals, yvals, color='k', marker='--', label='Deterministic')
    plt.xlabel('Number of Training pairs t')
    plt.ylabel('Average Risk')
    plt.title('Average Riks vs. Number of Training pairs')
    plt.legend()
    plt.show()


def plot_simple_mean_std(upper_std):
    with open('./experimental_results/exp_std_mean/result.json', 'r') as f:
        result = json.load(f)
        f.close()

    # color_palette = generate_color_palette(num_ranks)
    with open('./experimental_results/exp_std_mean/result.json' + 'config.json', 'r') as f:
        config = json.load(f)
    num_qbits = config['num_qbits']


    color = cm.rainbow(np.linspace(0, 10, upper_std))
    std_values = [i for i in range(0,upper_std)]
    for std, color in zip(std_values, color):
        plt.plot(std, result[std], color=color, label='std=' + str(std))

    # create deterministic line
    def f(t):
        return (1 - 1 / (2 ** num_qbits)) * (1 - t / (2 ** num_qbits))

    #xvals = np.linspace(0, 2 ** num_qbits, 200)
    #yvals = list(map(f, xvals))
    #plt.plot(xvals, yvals, color='k', marker='--', label='Deterministic')
    plt.xlabel('Standard deviation')
    plt.ylabel('Average Risk')
    plt.title('Average Riks vs. Std')
    plt.legend()
    plt.show()


def plot_mean_std():
    with open('./experimental_results/exp_std_mean/result.json', 'r') as f:
        result = json.load(f)
        f.close()

    # color_palette = generate_color_palette(num_ranks)
    with open('./experimental_results/exp_std_mean/result.json' + 'config.json', 'r') as f:
        config = json.load(f)
    num_qbits = config['num_qbits']


    # color_palette = generate_color_palette(num_ranks)

    max_rank = 2**config['num_qbits']
    color = cm.rainbow(np.linspace(0, 10, max_rank- config['rank']))
    markers = generate_markers()
    num_train_pairs = [i for i in range(1, 2 ** max_rank + 1)]

    for std, color in zip(ranks, color):
        plt.plot(num_train_pairs, result[std], color=color, marker=markers[ra], label='std=' + str(std))

    # create deterministic line
    def f(t):
        return (1 - 1 / (2 ** num_qbits)) * (1 - t / (2 ** num_qbits))

    #xvals = np.linspace(0, 2 ** num_qbits, 200)
    #yvals = list(map(f, xvals))
    #plt.plot(xvals, yvals, color='k', marker='--', label='Deterministic')
    plt.xlabel('Number of Training pairs t')
    plt.ylabel('Average Risk')
    plt.title('Average Riks vs. Number of Training pairs')
    plt.legend()
    plt.show()
    
    
def plot_loss(losses, num_qbits, num_layers, num_points, r_list, name_addition=''):
    if isinstance(losses[0], list):
        losses = np.array(losses)
        # losses shape is r x layer x epochs
        for j in range(len(num_layers)):
            num_layer = num_layers[j]
            for i in range(len(r_list)):
                r = r_list[i]
                loss = losses[i, j]
                plt.plot(list(range(len(loss))), loss, label=f"r={r}")
            plt.legend()
            plt.title(f"Loss for net with {num_qbits} qbits, {num_layer} layers, {num_points} data points")
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.savefig(f"./plots/loss_{num_qbits}_qbits_{num_layer}_layers_{num_points}_datapoints{name_addition}.png")
            plt.cla()
    else:
        plt.plot(list(range(len(losses))), losses)
        plt.title(f"Loss for net with {num_qbits} qbits, {num_layers} layers, {num_points} data points, {2**r_list} schmidt rank")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(f"./plots/loss_{num_qbits}_qbits_{num_layers}_layers_{num_points}_datapoints_{2**r_list}_schmidtrank{name_addition}.png")
        plt.cla()


if __name__ == '__main__':
    
