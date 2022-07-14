from utils import uniform_random_data, random_unitary_matrix, torch_tensor_product
import time
import numpy as np
from utils import quantum_risk
import torch
import matplotlib.pyplot as plt
from typing import List
import importlib

torch.manual_seed(4241)
np.random.seed(4241)


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


def quick_matmulvec(M, vec):
    size = M.shape[0]
    result = torch.zeros(vec.shape, dtype=torch.complex128)
    for i in range(0, vec.shape[0], size):
        result[i:i+size] = torch.matmul(M, vec[i:i+size])
    return result


def quick_matmulmat(A, B):
    size = B.shape[0]
    result = torch.zeros(A.shape, dtype=torch.complex128)
    for i in range(0, A.shape[0], size):
        for j in range(0, A.shape[1], size):
            result[i:i+size, j:j+size] = torch.matmul(A[i:i+size, j:j+size], B)
    return result


def cost_func(X, y_conj, qnn):
    cost = torch.zeros((1,))
    V = qnn.get_tensor_V()
    for idx in range(len(X)):
        state = quick_matmulvec(V, X[idx])
        state = torch.dot(y_conj[idx], state)
        cost += torch.square(state.real) + torch.square(state.imag)
        # cost += torch.square(torch.abs(torch.dot(el, state)))
    cost /= len(X)
    return 1 - cost


def train(X, y_conj, qnn, num_epochs, optimizer, scheduler=None):
    losses = []
    for i in range(num_epochs):
        loss = cost_func(X, y_conj, qnn)
        losses.append(loss.item())
        if i % 100 == 0:
            print(f"\tepoch [{i+1}/{num_epochs}] loss={loss.item()}")
        if loss.item() == 0.0:
            # print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}\nstopped")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0 and scheduler is not None:
            scheduler.step()
            print(f"\tepoch [{i+1}/{num_epochs}] lr={scheduler.get_lr()}")
    print(f"\tepoch [{num_epochs}/{num_epochs}] final loss {losses[-1]}")
    return losses


def I(size):
    matrix = torch.zeros((size, size), dtype=torch.complex128)
    for i in range(size):
        matrix[i, i] = 1
    return matrix


def init(num_layers, num_qbits, schmidt_rank, num_points, num_epochs, lr, qnn_name, opt_name='Adam'):
    starting_time = time.time()
    x_qbits = num_qbits
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    x_wires = list(range(num_qbits))
    qnn = getattr(importlib.import_module('qnn'), qnn_name)(wires=x_wires, num_layers=num_layers, use_torch=True)

    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits, r_first=True)))
    U = random_unitary_matrix(x_qbits)
    # U = qnn.get_matrix_V()
    # U_inv = U.conj().T
    # r_I = I(2**r_qbits)
    # U_inv_I = torch_tensor_product(torch.from_numpy(U_inv), r_I)
    # X[i, :] is the i'th input => X contains them in as row vectors
    # U*X.T gives us y, but as column vectors
    # (U*X.T).T gives us y, as row vectors and (U*X.T).T = X*U.T
    y_conj = quick_matmulmat(X, torch.from_numpy(U.T)).conj()

    if opt_name.lower() == 'sgd':
        optimizer = torch.optim.SGD
    else:
        optimizer = torch.optim.Adam

    if isinstance(qnn.params, list):
        optimizer = optimizer(qnn.params, lr=lr)
    else:
        optimizer = optimizer([qnn.params], lr=lr)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1)
    prep_time = time.time() - starting_time
    print(f"\tPreparation with {num_qbits} qubits and {num_layers} layers took {prep_time}s")

    starting_time = time.time()
    losses = train(X, y_conj, qnn, num_epochs, optimizer, scheduler)
    train_time = time.time() - starting_time

    print(f"\trisk = {quantum_risk(U, qnn.get_matrix_V())}")
    return train_time, prep_time, losses


def plot_runtime_to_schmidt_rank():
    # num_layers = [1] + list(range(5, 20, 5))
    num_layers = [1, 2]
    qbits = [4]
    num_epochs = 200
    lr = 0.1
    # qnn = 'Circuit6QNN'
    qnn = 'PennylaneQNN'
    opt_name = 'Adam'
    for i in range(len(qbits)):
        qbit = qbits[i]
        r_list = [i for i in range(qbit+1)]
        num_points = 2**(qbit+1)
        train_times_r = []
        prep_times_r = []
        min_losses_r = []
        losses_r = []
        for j in range(len(r_list)):
            r = r_list[j]
            schmidt_rank = 2**r
            min_losses = []
            train_times = []
            prep_times = []
            losses_layer = []
            for k in range(len(num_layers)):
                num_layer = num_layers[k]
                print(f"\nStart training: qbit [{i+1}/{len(qbits)}], r [{j+1}/{len(r_list)}], layers [{k+1}/{len(num_layers)}]")
                training_time, prep_time, losses = init(num_layer, qbit, schmidt_rank, num_points, num_epochs, lr, qnn, opt_name=opt_name)
                losses_layer.append(losses)
                print(f"\tTraining with {qbit} qubits, {num_layer} layers and r={r} took {training_time}s\n")
                min_losses.append(np.array(losses).min())
                train_times.append(training_time)
                prep_times.append(prep_time)
            losses_r.append(losses_layer)
            min_losses_r.append(min_losses)
            train_times_r.append(train_times)
            prep_times_r.append(prep_times)

        plot_loss(losses_r, qbit, num_layers, num_points, r_list, name_addition=f"_{num_epochs}_epochs_lr={lr}_{qnn}")

        for i in range(len(r_list)):
            r = r_list[i]
            min_losses = min_losses_r[i]
            plt.plot(num_layers, min_losses, label=f"r={r}")
        plt.legend()
        plt.xlabel('number of layers')
        plt.ylabel('min loss')
        plt.title(f'Minimal loss in {num_epochs} epochs for a {qbit} qubit system')
        plt.savefig(f'./plots/minimal_loss_{qbit}_qbits_{num_epochs}_epochs_lr={lr}_{qnn}.png')
        plt.cla()

        for i in range(len(r_list)):
            r = r_list[i]
            times = train_times_r[i]
            plt.plot(num_layers, times, label=f"r={r}")
        plt.legend()
        plt.xlabel('number of layers')
        plt.ylabel('Time for Training [s]')
        plt.title(f'Time for Training for {num_epochs} epochs for a {qbit} qubit system')
        plt.savefig(f'./plots/train_time_{qbit}_qbits_{num_epochs}_epochs_lr={lr}_{qnn}.png')
        plt.cla()

        for i in range(len(r_list)):
            r = r_list[i]
            times = prep_times_r[i]
            plt.plot(num_layers, times, label=f"r={r}")
        plt.legend()
        plt.xlabel('Number of Layers')
        plt.ylabel('Time for Preparations [s]')
        plt.title(f'Time for Preparations for a {qbit} qubit system')
        plt.savefig(f'./plots/prep_time_{qbit}_qbits_lr={lr}_{qnn}.png')
        plt.cla()


def rough_train_time_requirements(exp_duration_s=0, exp_duration_min=0, exp_duration_hour=0, exp_duration_day=0):
    exp_time = 0
    exp_time += exp_duration_s
    exp_time += exp_duration_min*60
    exp_time += exp_duration_hour*60*60
    exp_time += exp_duration_day*60*60*24
    print(exp_time)

    # exp_time = t*num_epochs*7*10*100   #schimdrank*uniatires*datasets
    train_time = exp_time/7000.
    print(f"training may take up to {train_time}s")


def main():
    plot_runtime_to_schmidt_rank()
    # rough_train_time_requirements(0, 0, 0, 4)


if __name__ == '__main__':
    main()
