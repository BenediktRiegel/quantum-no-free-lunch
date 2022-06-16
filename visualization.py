import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from typing import List
import numpy as np
from evaluation import *
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


"""
def plot(max_rank):
    #generate pairs on which to get results
    ranks, num_train_pairs = generate_pairs(max_rank)

    #get results of experiment
    data_points = get_result(ranks,num_train_pairs)

    #color_palette = generate_color_palette(num_ranks)
    color = cm.rainbow(np.linspace(0, 10, max_rank))
    markers = generate_markers()


    for rank, color in zip(ranks, color):
            plt.plot(data_points[rank][0], data_points[rank][1], color=color, marker = markers[rank] ,label='r='+ str(rank))

    #create deterministic line
    def f(t):
        return (1-1/max_rank)*(1-t/max_rank)



    xvals = np.linspace(0,max_rank,200)
    yvals = list(map(f, xvals))
    plt.plot(xvals, yvals, color='k', marker='--',label= 'Deterministic')
    plt.xlabel('Number of Training pairs t')
    plt.ylabel('Average Risk')
    plt.title('Average Riks vs. Number of Training pairs')
    plt.legend()
    plt.show()
"""
