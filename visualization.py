
#create plots fo training data
# lower

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from typing import List
import numpy as np
from evaluation import *


#create plot analogous average risk vs. training pairs


def set_fontsize(text=10, title=10, labels=10, xtick=10, ytick=10, legend=10):
    plt.rc('font', size=text)  # controls default text size
    plt.rc('axes', titlesize=title)  # fontsize of the title
    plt.rc('axes', labelsize=labels)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=xtick)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=ytick)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=legend)  # fontsize of the legend


def generate_markers():
    markers = [".", ",", "o", "v","^","<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_"]


def generate_pairs(max_rank):
    #construct pairs on which to evaluate experiment
    ranks = [2**i for i in range(0, max_rank)]    
    num_train_pairs= [i for i in range(0,max_rank)]
    return ranks, num_train_pairs

def get_result(ranks, num_train_pairs):
     data_points = get_test_results(ranks, num_train_pairs)
     return data_points

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
