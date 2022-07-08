from qnn import QNN, PennylaneQNN
import pennylane as qml
from quantum_backends import QuantumBackends
from utils import uniform_random_data, random_unitary_matrix, torch_tensor_product
import time
import numpy as np
from typing import List
from utils import abs2, quantum_risk
import torch

torch.manual_seed(4241)
np.random.seed(4241)


def cost_func(X, qnn, unitary, r_I):
    cost = torch.zeros((1,))
    V = torch_tensor_product(qnn.get_tensor_V(), r_I)
    for el in X:
        state = torch.matmul(V, el)
        state = torch.matmul(unitary, state)
        cost += torch.square(torch.abs(torch.dot(el, state)))
    cost /= len(X)
    return 1 - cost


def train(X, qnn, unitary, num_epochs, optimizer, r_I):
    for i in range(num_epochs):
        loss = cost_func(X, qnn, unitary, r_I)
        if i % 20 == 0:
            print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}")
        if loss.item() == 0.0:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def I(size):
    matrix = torch.zeros((size, size), dtype=torch.complex128)
    for i in range(size):
        matrix[i, i] = 1
    return matrix


def init(num_layers, num_qbits):
    x_qbits = num_qbits
    r_qbits = num_qbits
    x_wires = list(range(num_qbits))
    qnn = PennylaneQNN(wires=x_wires, num_layers=num_layers, use_torch=True)

    print('prep')
    X = torch.from_numpy(np.array(uniform_random_data(1, 2, x_qbits, r_qbits)))
    U = random_unitary_matrix(x_qbits)
    U_inv = U.conj().T
    r_I = I(2**r_qbits)
    U_inv_I = torch_tensor_product(torch.from_numpy(U_inv), r_I)

    optimizer = torch.optim.SGD([qnn.params], lr=0.1)

    starting_time = time.time()
    train(X, qnn, U_inv_I, 1, optimizer, r_I)
    total_time = time.time() - starting_time

    print(quantum_risk(U, qnn.get_matrix_V()))
    return total_time


def main():
    num_layers = 1
    for i in range(1, 7):
        training_time = init(num_layers, i)


if __name__ == '__main__':
    main()
