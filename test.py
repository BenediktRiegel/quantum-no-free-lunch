import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import List
from utils import *
from qnn import *
import matplotlib.pyplot as plt

"""
def construct_circuit(trained_params, num_layers, x_qbits):
    qnn = PennylaneQNN(wires=list(range(x_qbits)), num_layers=num_layers)
    return qnn.qnn()
"""


def calc_risk_qnn(trained_qnn, U):

    #circuit = construct_circuit(trained_params, num_layers, x_qbits)
    V = trained_qnn.get_matrix_V()
    risk = quantum_risk(U, V)
    return risk


def calc_avg_risk(schmidt_rank, num_points, x_qbits, r_qbits, num_unitaries, num_layers):
    sum_risk = 0
    for i in range(num_unitaries):
        unitary = random_unitary_matrix(x_qbits)
        dataset = SchmidtDataset(schmidt_rank, num_points, x_qbits, r_qbits)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        qnn = PennylaneQNN(wires=list(range(x_qbits)), num_layers=num_layers)

        ref_wires = list(range(x_qbits, x_qbits+r_qbits))
        dev = qml.device('default.qubit', wires=qnn.wires+ref_wires)
        losses = train_qnn(qnn, unitary, dataloader, ref_wires, dev)
        plt.plot(list(range(len(losses))), losses)
        trained_params = qnn.params
        risk =calc_risk_qnn(qnn, unitary)
        sum_risk += risk
    plt.grid(True)
    plt.show()
    average_risk = sum_risk/num_unitaries
    return average_risk


def main():
    params = {
        "schmidt_rank": 2,
        "num_points": 3,
        "x_qbits":  2,
        "r_qbits": 2,
        "num_unitaries": 4,
        "num_layers": 10
    }
    print(calc_avg_risk(**params))


if __name__ == '__main__':
    main()
