import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from scipy.stats import unitary_group
from scipy.optimize import fmin_cobyla
from utils import *
from typing import List




def uniformly_sample_from_base(num_qbits: int, size: int):
    """
    Draws vectors from an orthonormal basis (generated from the standard basis
    by multiplication with a random unitary)

    Parameters
    ----------
    num_qbits : int
        number of qubits of the input space -> dimension is 2**num_qbits
    size : int
        number of basis vectors to be drawn
    """
    if num_qbits == 0:
        return np.ones((1, 1))
    # uniform sampling of basis vectors
    num_bits = np.power(2, num_qbits)
    base = []
    random_ints = np.random.choice(num_bits, size, replace=False)
    transform_matrix = unitary_group.rvs(num_bits)
    for rd_int in range(len(random_ints)):
        binary_base = one_hot_encoding(random_ints[rd_int], num_bits)
        base.append(binary_base)

    return np.array(base) @ transform_matrix

def uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits):
    """
    Generates a random point with a specified schmidt rank by drawing basis vectors corresponding
    to the schmidt rank and 'pairing' them in a linear combination of elementary tensors

    Parameters
    ----------
    schmidt_rank : int
        determines how many basis vectors are drawn for the circuit and the reference system
    x_qbits, r_qbits: int
        specify the amount of qubits in the circuit and reference system
    """
    basis_x = uniformly_sample_from_base(x_qbits, schmidt_rank)
    basis_r = uniformly_sample_from_base(r_qbits, schmidt_rank)
    coeff = np.random.uniform(size=schmidt_rank)
    point = np.zeros((2**x_qbits * 2**r_qbits), dtype=np.complex128)
    for i in range(schmidt_rank):
        point += coeff[i] * tensor_product(basis_r[i], basis_x[i])
    return normalize(point)



def uniform_random_data(schmidt_rank: int, size: int, x_qbits: int, r_qbits: int) -> List[List[float]]:
    """
    Generates a data set of specified size with a given schmidt rank by drawing points
    with uniformly_sample_random_point

    Parameters
    ----------
    schmidt_rank : int
        Desired Schmidt rank of the points in the data set
    size : int
        Desired size of the data set (number of points)
    x_qbits, r_qbits : int
        Desired input size of the circuit and reference system
    """
    data = []
    # size = number data samples of trainset
    for i in range(size):
        data.append(uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits))
    return data



def create_mean_std(mean, std, num_samples, max_rank, counter):
    """
    (Approximately) generates a set of integers in a specified range with certain mean and standard deviation

    Parameters
    ----------
    mean, std : int
        Desired mean and standard deviation of the sampled integers
    num_samples: int
        Desired amount of integers
    max_rank : int
        Maximal allowed integer to be drawn
    counter : int
        helper parameter to analyze rejection rate
    """
    data = []

    min_dist = min(mean - 1, max_rank - mean)
    if(min_dist <= 3 * std):
        raise ValueError(f'Bad standard deviation')
    samples = np.random.normal(loc=0.0, scale= std, size=num_samples)

    #samples = np.random.randint(mean - 5 * std, mean + 5 * std, size=num_samples)

    print('testing of mean_std data generation')
    actual_mean = np.mean(samples)
    actual_std = np.std(samples)
    #print(samples)
    print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))
    zero_mean_samples = samples - (actual_mean)

    zero_mean_mean = np.mean(zero_mean_samples)
    zero_mean_std = np.std(zero_mean_samples)
    #print(zero_mean_samples)
    print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))

    scaled_samples = zero_mean_samples * (std/zero_mean_std)
    scaled_mean = np.mean(scaled_samples)
    scaled_std = np.std(scaled_samples)
    #print(scaled_samples)
    print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))

    final_samples = scaled_samples + mean
    final_samples = np.round_(final_samples)
    final_mean = np.mean(final_samples)
    final_std = np.std(final_samples)
    print(final_samples)
    print("Final samples stats     : mean = {:.4f} stdv = {:.4f}".format(final_mean, final_std))

    if any(number <= 0 or number > max_rank for number in final_samples):
        counter = counter + 1
        final_samples , counter = create_mean_std(mean, std, num_samples, max_rank, counter)

    return final_samples, counter


#create dataset of size <size> with a mean schmidt rank
def uniform_random_data_mean(mean, std, num_samples, x_qbits, r_qbits):
    """
    Create dataset of specified size with variable Schmidt rank with certain mean and standard
    deviation

    Parameters
    ----------
    mean, std : int
        Desired mean and standard deviation of the Schmidt ranks within the data set
    num_samples : int
        Desired size of the data set
    x_qbits, r_qbits :
        Desired input size of circuit and reference system
    """
    data = []
    numbers_mean_std, counter = create_mean_std(mean, std, num_samples)
    for i in range(len(numbers_mean_std)):
        schmidt_rank = numbers_mean_std[i]
        data.append(uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits))
    return data


def random_unitary_matrix(x_qbits):
    """
    Generates Haar-distributed unitary

    Parameters
    ----------
    x_qbits : int
        Dimension of input system -> unitary has shape (2**x_qbits, 2**x_qbits)
    """
    matrix = unitary_group.rvs(2**x_qbits)
    return matrix



class SchmidtDataset(torch.utils.data.Dataset):
    def __init__(self, schmidt_rank, num_points, x_qbits, r_qbits):
        # Initialize the data and label list
        self.data = uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)
        self.labels = [0]*num_points

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.data)


class SchmidtDataset_std(torch.utils.data.Dataset):
    def __init__(self, schmidt_rank, num_points, x_qbits, r_qbits, std):
        # Initialize the data and label list
        self.data = uniform_random_data_mean(schmidt_rank, std, num_points, x_qbits, r_qbits)
        self.labels = [0]*num_points

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.data)
