import time
import numpy as np
from metrics import quantum_risk
import torch
import matplotlib.pyplot as plt
import importlib
from classic_training import train
from classic_training import init as init_classic_training
from data import uniform_random_data, random_unitary_matrix

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
            # plt.yscale('log')
            # plt.xscale('log')
            plt.title(f"Loss for net with {num_qbits} qbits, {num_layer} layers, {num_points} data points")
            plt.xlabel("epochs")
            plt.ylabel("loss")
            plt.savefig(f"./plots/loss_{num_qbits}_qbits_{num_layer}_layers_{num_points}_datapoints{name_addition}.png")
            plt.cla()
    else:
        plt.plot(list(range(len(losses))), losses)
        # plt.yscale('log')
        # plt.xscale('log')
        plt.title(f"Loss for net with {num_qbits} qbits, {num_layers} layers, {num_points} data points, {2**r_list} schmidt rank")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(f"./plots/loss_{num_qbits}_qbits_{num_layers}_layers_{num_points}_datapoints_{2**r_list}_schmidtrank{name_addition}.png")
        plt.cla()


def init(num_layers, num_qbits, schmidt_rank, num_points, num_epochs, lr, qnn_name, opt_name='Adam', device='cpu'):
    """
        Tensor training for QNN
        """
    starting_time = time.time()
    x_qbits = num_qbits
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))  # dont use all qubits for reference system
    x_wires = list(range(num_qbits))  # does not matter which qubits we are using, since we only want the matrix

    # construct QNNobject from qnn_name string
    if 'cuda' in qnn_name.lower():
        qnn = getattr(importlib.import_module('cuda_qnn'), qnn_name)(num_wires=len(x_wires), num_layers=num_layers, device=device)
    else:
        qnn = getattr(importlib.import_module('qnn'), qnn_name)(wires=x_wires, num_layers=num_layers, use_torch=True)

    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits))).to(device)

    U = torch.tensor(random_unitary_matrix(x_qbits), device=device)

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, min_lr=1e-10, verbose=True)
    prep_time = time.time() - starting_time
    print(f"\tPreparation with {num_qbits} qubits and {num_layers} layers took {prep_time}s")

    init_classic_training(device=device)
    starting_time = time.time()
    losses = train(X, U, qnn, num_epochs, optimizer, scheduler, device=device)
    train_time = time.time() - starting_time

    print(f"\trisk = {quantum_risk(U, qnn.get_matrix_V())}")
    return train_time, prep_time, losses


def min_loss_over_layer(r_list, min_losses_r, num_layers, num_epochs, qbit, lr, qnn):
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


def train_time_over_num_layer(r_list, train_times_r, num_layers, num_epochs, qbit, lr, qnn):
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


def plot_runtime_to_schmidt_rank():
    # num_layers = [1] + list(range(5, 20, 5))
    num_layers = [10]
    qbits = [6]
    num_epochs = 200
    lr = 0.1
    # qnns = ['PennylaneQNN', 'OffsetQNN', 'Circuit2QNN', 'Circuit5QNN', 'Circuit6QNN', 'Circuit9QNN']
    # qnns = ['Circuit11QNN', 'Circuit12QNN', 'Circuit13QNN', 'Circuit14QNN']
    qnns = ['CudaCircuit6']
    qnns = ['CudaEfficient']
    qnns = ['CudaPennylane']
    # qnns = ['CudaSimpleEnt']
    # qnns = ['CudaComplexPennylane']
    device = 'cpu'
    # device = 'cuda:0'
    opt_name = 'Adam'
    qnn_losses = []
    qnn_times = []
    qnn_idx = 0
    for qnn in qnns:
        qnn_idx += 1
        for i in range(len(qbits)):
            qbit = qbits[i]
            # r_list = [i for i in range(qbit+1)]
            r_list = [qbit]
            num_points = 2**(qbit)
            # num_points = 1
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
                    training_time, prep_time, losses = init(num_layer, qbit, schmidt_rank, num_points, num_epochs, lr, qnn, opt_name=opt_name, device=device)
                    losses_layer.append(losses)
                    print(f"\tTraining with {qbit} qubits, {num_layer} layers and r={r} took {training_time}s\n")
                    min_losses.append(np.array(losses).min())
                    train_times.append(training_time)
                    prep_times.append(prep_time)
                    plot_loss(losses, qbit, num_layer, num_points, r, name_addition=f"_{qnn}")

                losses_r.append(losses_layer)
                min_losses_r.append(min_losses)
                train_times_r.append(train_times)
                prep_times_r.append(prep_times)

                qnn_losses.append((qnn, r, num_points, min_losses))
                qnn_times.append((qnn, r, num_points, train_times))

        # plot_loss(losses_r, qbit, num_layers, num_points, r_list, name_addition=f"_{num_epochs}_epochs_lr={lr}_{qnn}")
    for i in range(len(qnn_losses)):
        qnn, r, d, loss = qnn_losses[i]
        plt.plot(num_layers, loss, label=f"{qnn}, r={r}, d={d}")
    plt.legend()
    plt.xlabel('number of layers')
    plt.ylabel('min loss')
    plt.savefig('./plots/qnn_min_loss.png')
    plt.cla()
    for i in range(len(qnn_times)):
        qnn, r, d, time = qnn_times[i]
        plt.plot(num_layers, time, label=f"{qnn}, r={r}, d={d}")
    plt.legend()
    plt.xlabel('number of layers')
    plt.ylabel('time in s')
    plt.savefig('./plots/qnn_time.png')
    plt.cla()


def main():
    plot_runtime_to_schmidt_rank()
    # rough_train_time_requirements(0, 0, 0, 4)


if __name__ == '__main__':
    main()
