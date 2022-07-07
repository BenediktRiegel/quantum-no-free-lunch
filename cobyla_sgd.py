from scipy.optimize import minimize
from qnn import PennylaneQNN, QNN, cost_func, SchmidtDataset, DataLoader
from test import calc_risk_qnn
from typing import List
import pennylane as qml
from utils import adjoint_unitary_circuit, random_unitary_matrix, uniform_random_data
from config import gen_config
import numpy as np
from experiment import Writer
import torch
from log import Logger
import time


def get_cost_func(X_train, qnn: QNN, unitary, ref_wires: List[int], dev: qml.Device):
    def preped_cost_func(qnn_params: np.ndarray):
        # print(qnn.params.shape)
        # qnn_params.reshape()
        qnn.params = qnn_params.reshape(qnn.params.shape)
        # print(qnn_params)
        cost = 0
        for el in X_train:
            @qml.qnode(dev)
            def circuit():
                # print(f"all wires {qnn.wires+ref_wires}")
                # print(f"{len(el)} should equal 2**{len(qnn.wires)+len(ref_wires)} = {2**(len(qnn.wires)+len(ref_wires))}")
                qml.MottonenStatePreparation(el, wires=qnn.wires + ref_wires)  # Amplitude Encoding
                qnn.qnn()
                adjoint_unitary_circuit(unitary)(wires=qnn.wires)  # Adjoint U
                qml.MottonenStatePreparation(el, wires=qnn.wires + ref_wires).inv()  # Inverse Amplitude Encoding
                return qml.probs(wires=qnn.wires + ref_wires)

            # print('run circuit')
            cost += circuit()[0]

        return 1 - (cost / len(X_train))

    return preped_cost_func


def train_qnn(qnn: QNN, unitary, dataset, ref_wires: List[int],
              dev: qml.Device, num_epochs: int, learning_rate: float,
              dataloader):
    # num_qubits = len(qnn.wires) + len(ref_wires)
    # num_layers = qnn.num_layers
    steps = num_epochs
    # all_losses = []
    f = get_cost_func(dataset, qnn, unitary, ref_wires, dev)
    start_time = time.time()
    result = minimize(f, x0=qnn.params.copy().flatten(), method='COBYLA')
    total_time = time.time() - start_time
    print(f"Cobyla training took {total_time}s")

    start_time = time.time()
    qnn_torch_params = torch.autograd.Variable(torch.tensor(qnn.params), requires_grad=True)
    qnn.params = qnn_torch_params
    # set up the optimizer
    # opt = torch.optim.Adam([qnn.params], lr=learning_rate)
    opt = torch.optim.SGD([qnn.params], lr=learning_rate)

    # optimization begins
    all_losses = []
    for n in range(steps):
        print(f"step {n + 1}/{steps}")
        opt.zero_grad()
        total_loss = 0
        for X in dataloader:
            print('calc cost funktion')
            loss = cost_func(X[0], qnn, unitary, ref_wires, dev)
            print('backprop')
            loss.backward()
            print('optimise')
            opt.step()
            print('total loss')
            total_loss += loss.item()

        all_losses.append(total_loss)
        # Keep track of progress every 10 steps
        if n % 10 == 9 or n == steps - 1:
            print(f"Cost after {n + 1} steps is {total_loss}")
        if total_loss == 0.0:
            print(f"loss({total_loss}) = 0.0")
            break
    total_time = time.time() - start_time
    print(f"Gradient training took {total_time}s")
    print(all_losses)


def avg_risk(schmidt_rank, num_points, x_qbits, r_qbits,
             num_unitaries, num_layers, num_training_data,
             num_epochs, learning_rate, batch_size, logger):
    sum_risk = 0
    for i in range(num_unitaries):
        # Draw random unitary
        # print(f"unitary {i + 1}/{num_unitaries}")
        logger.update_num_unitary(i)
        unitary = random_unitary_matrix(x_qbits)
        # Train it with <num_train_data> many random datasets
        for j in range(num_training_data):
            # print(f"training dataset {j + 1}/{num_training_data}")
            logger.update_num_training_dataset(j)
            # Init data and neural net
            dataset = uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)
            qnn = PennylaneQNN(wires=list(range(x_qbits)), num_layers=num_layers, use_torch=False)
            # Init quantum device
            ref_wires = list(range(x_qbits, x_qbits + r_qbits))
            dataset = SchmidtDataset(schmidt_rank, num_points, x_qbits, r_qbits)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            dev = qml.device('lightning.qubit', wires=qnn.wires + ref_wires)
            dev.shots = 1000
            # Train and compute risk
            print('training qnn')
            train_qnn(qnn, unitary, dataset, ref_wires, dev, num_epochs, learning_rate, dataloader)
            # plt.plot(list(range(len(losses))), losses)
            print('calculating risk')
            risk = calc_risk_qnn(qnn, unitary)
            sum_risk += risk
    # plt.grid(True)
    # plt.show()

    # average over all unitaries and training sets
    print('average risk')
    average_risk = sum_risk / (num_unitaries * num_training_data)
    return average_risk


def exp_fig2_3(config, save_dir):
    writer = Writer(save_dir + 'cobyla_sgd_result.txt')
    logger = Logger(config['x_qbits'], config['num_points'], config['num_unitaries'],
                    config['num_training_data'])
    all_risks = []
    for i in range(config['x_qbits'] + 1):
        rank = 2 ** i
        risk_list = []
        # print(f"rank {i + 1}/{config['x_qbits'] + 1} (rank={rank})")
        r_qbits = i
        logger.update_schmidt_rank(i)
        for num_points in range(1, config['num_points'] + 1):
            # print(f"num_points {num_points}/{config['num_points']}")
            logger.update_num_points(num_points)
            risk = avg_risk(rank, num_points, config['x_qbits'],
                            r_qbits, config['num_unitaries'],
                            config['num_layers'], config['num_training_data'],
                            config['num_epochs'], config['learning_rate'],
                            config['batch_size'], logger
                            )
            writer.append(risk)
            risk_list.append(risk)
        all_risks.append(risk_list)
    all_risks_array = np.array(all_risks)

    # store risks
    np.save(save_dir + 'cobyla_sgd_result.npy', all_risks_array)


def fig2():
    save_dir = './experimental_results/exp1/'
    config = gen_config(2, 2, 1, 1, 10, 4, 10, 0, num_epochs=1000)
    exp_fig2_3(config, save_dir)


def main():
    fig2()


if __name__ == '__main__':
    main()
