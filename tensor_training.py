from qnn import QNN, PennylaneQNN, OffsetQNN
from utils import uniform_random_data, random_unitary_matrix, torch_tensor_product
import time
import numpy as np
from utils import quantum_risk
import torch
import matplotlib.pyplot as plt
from typing import List

torch.manual_seed(4241)
np.random.seed(4241)


def plot_loss(losses, num_qbits, num_layers, num_points, schmidt_rank):
    if isinstance(losses[0], list):
        losses = np.array(losses)
        # losses shape is r x layer x epochs
        for j in range(len(num_layers)):
            num_layer = num_layers[j]
            for i in range(len(schmidt_rank)):
                r = schmidt_rank[i]
                loss = losses[i, j]
                plt.plot(list(range(len(loss))), loss, label=f"r={r}")
            plt.legend()
            plt.title(f"Loss for net with {num_qbits} qbits, {num_layer} layers, {num_points} data points")
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.savefig(f"./plots/loss_{num_qbits}_qbits_{num_layer}_layers_{num_points}_datapoints.png")
            plt.cla()
    else:
        plt.plot(list(range(len(losses))), losses)
        plt.title(f"Loss for net with {num_qbits} qbits, {num_layers} layers, {num_points} data points, {schmidt_rank} schmidt rank")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(f"./plots/loss_{num_qbits}_qbits_{num_layers}_layers_{num_points}_datapoints_{schmidt_rank}_schmidtrank.png")
        plt.cla()


def cost_func(X, qnn, unitary, r_I):
    cost = torch.zeros((1,))
    V = torch_tensor_product(qnn.get_tensor_V(), r_I)
    for el in X:
        state = torch.matmul(V, el)
        state = torch.matmul(unitary, state)
        state = torch.dot(torch.conj(el), state)
        cost += torch.square(state.real) + torch.square(state.imag)
        # cost += torch.square(torch.abs(torch.dot(el, state)))
    cost /= len(X)
    return 1 - cost


def train(X, qnn, unitary, num_epochs, optimizer, r_I):
    losses = []
    for i in range(num_epochs):
        loss = cost_func(X, qnn, unitary, r_I)
        losses.append(loss.item())
        # if i % 20 == 0:
        #     print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}")
        if loss.item() == 0.0:
            # print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}\nstopped")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch [{i+1}/{num_epochs}] final loss {losses[-1]}")
    return losses


def I(size):
    matrix = torch.zeros((size, size), dtype=torch.complex128)
    for i in range(size):
        matrix[i, i] = 1
    return matrix


def init(num_layers, num_qbits, schmidt_rank, num_points, num_epochs):
    x_qbits = num_qbits
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    x_wires = list(range(num_qbits))
    qnn = PennylaneQNN(wires=x_wires, num_layers=num_layers, use_torch=True)

    print('prep')
    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
    U = random_unitary_matrix(x_qbits)
    # U = qnn.get_matrix_V()
    U_inv = U.conj().T
    r_I = I(2**r_qbits)
    U_inv_I = torch_tensor_product(torch.from_numpy(U_inv), r_I)

    optimizer = torch.optim.Adam([qnn.params], lr=0.1)

    starting_time = time.time()
    losses = train(X, qnn, U_inv_I, num_epochs, optimizer, r_I)
    total_time = time.time() - starting_time

    print(f"risk = {quantum_risk(U, qnn.get_matrix_V())}")
    return total_time, losses


def plot_runtime_to_num_layers():
    # time = init(num_layers=1, num_qbits=1)
    # print(f"{time}s")
    # num_layers = [1] + list(range(5, 100, 5))
    # qbits = [4]
    num_epochs = 200
    num_points = 2
    num_layers = [1]
    qbits = [1]
    schmidt_rank = 1
    min_losses = []
    times = []
    for qbit in qbits:
        for num_layer in num_layers:
            training_time, losses = init(num_layer, qbit, 1, num_points, num_epochs)
            plot_loss(losses, qbit, num_layer, num_points, schmidt_rank)
            print(f"Training with {qbit} qubits and {num_layer} layers took {training_time}s")
            min_losses.append(np.array(losses).min())
            times.append(training_time)
        plt.plot(num_layers, min_losses)
        plt.xlabel('number of layers')
        plt.ylabel('min loss')
        plt.title(f'Minimal loss in {num_epochs} epochs for a {qbit} qubit system')
        plt.savefig('./plots/minimal_loss.png')
        plt.cla()
        plt.plot(num_layers, times)
        plt.xlabel('number of layers')
        plt.ylabel('runtime [s]')
        plt.title(f'Run time of {num_epochs} epochs for a {qbit} qubit system')
        plt.savefig('./plots/runtime.png')
        plt.cla()


def plot_runtime_to_schmidt_rank():
    num_layers = [1, 5] + list(range(10, 100, 10))
    qbits = [5]
    num_epochs = 200
    for i in range(len(qbits)):
        qbit = qbits[i]
        schmidt_rank = [2**i for i in range(qbit+1)]
        num_points = 2**(qbit+1)
        times_r = []
        min_losses_r = []
        losses_r = []
        for j in range(len(schmidt_rank)):
            r = schmidt_rank[j]
            min_losses = []
            times = []
            losses_layer = []
            for k in range(len(num_layers)):
                num_layer = num_layers[k]
                print(f"Start training: qbit [{i+1}/{len(qbits)}], r [{j+1}/{len(schmidt_rank)}], layers [{k+1}/{len(num_layers)}]")
                training_time, losses = init(num_layer, qbit, r, num_points, num_epochs)
                losses_layer.append(losses)
                print(f"Training with {qbit} qubits, {num_layer} layers and r={r} took {training_time}s")
                min_losses.append(np.array(losses).min())
                times.append(training_time)
            losses_r.append(losses_layer)
            min_losses_r.append(min_losses)
            times_r.append(times)

        plot_loss(losses_r, qbit, num_layers, num_points, schmidt_rank)

        for i in range(len(schmidt_rank)):
            r = schmidt_rank[i]
            min_losses = min_losses_r[i]
            plt.plot(num_layers, min_losses, label=f"r={r}")
        plt.legend()
        plt.xlabel('number of layers')
        plt.ylabel('min loss')
        plt.title(f'Minimal loss in {num_epochs} epochs for a {qbit} qubit system')
        plt.savefig(f'./plots/minimal_loss_{qbit}_qbits_{num_epochs}_epochs.png')
        plt.cla()
        for i in range(len(schmidt_rank)):
            r = schmidt_rank[i]
            times = times_r[i]
            plt.plot(num_layers, times, label=f"r={r}")
        plt.legend()
        plt.xlabel('number of layers')
        plt.ylabel('runtime [s]')
        plt.title(f'Run time of {num_epochs} epochs for a {qbit} qubit system')
        plt.savefig(f'./plots/runtime_{qbit}_qbits_{num_epochs}_epochs.png')
        plt.cla()


def main():
    plot_runtime_to_schmidt_rank()


if __name__ == '__main__':
    main()
