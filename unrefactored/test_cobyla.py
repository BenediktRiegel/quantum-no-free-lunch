from scipy.optimize import minimize
from qnn import PennylaneQNN, QNN
from test import calc_risk_qnn
from typing import List
import pennylane as qml
from utils import adjoint_unitary_circuit, random_unitary_matrix, uniform_random_data
from config import gen_config
import numpy as np
from experiment import Writer


def get_cost_func(X_train, qnn: QNN, unitary, ref_wires: List[int], dev: qml.Device):
    def cost_func(qnn_params: np.ndarray):
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

    return cost_func


def train_qnn(qnn: QNN, unitary, dataset, ref_wires: List[int],
              dev: qml.Device, num_epochs: int):
    # num_qubits = len(qnn.wires) + len(ref_wires)
    # num_layers = qnn.num_layers
    steps = num_epochs
    # all_losses = []
    f = get_cost_func(dataset, qnn, unitary, ref_wires, dev)
    result = minimize(f, x0=qnn.params.copy().flatten(), method='COBYLA')


def avg_risk(schmidt_rank, num_points, x_qbits, r_qbits,
            num_unitaries, num_layers, num_training_data,
            num_epochs):
    sum_risk = 0
    for i in range(num_unitaries):
        # Draw random unitary
        print(f"unitary {i + 1}/{num_unitaries}")
        unitary = random_unitary_matrix(x_qbits)
        # Train it with <num_train_data> many random datasets
        for j in range(num_training_data):
            print(f"training dataset {j + 1}/{num_training_data}")
            # Init data and neural net
            dataset = uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)
            qnn = PennylaneQNN(wires=list(range(x_qbits)), num_layers=num_layers, use_torch=False)
            # Init quantum device
            ref_wires = list(range(x_qbits, x_qbits + r_qbits))

            dev = qml.device('lightning.qubit', wires=qnn.wires + ref_wires)
            dev.shots = 1000
            # Train and compute risk
            print('training qnn')
            train_qnn(qnn, unitary, dataset, ref_wires, dev, num_epochs)
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


def main():
    save_dir = './experimental_results/exp1/'
    writer = Writer(save_dir + 'cobyla_result.txt')
    all_risks = []
    config = gen_config(2, 2, 1, 1, 10, 4, 10, 0)
    for i in range(config['x_qbits'] + 1):
        rank = 2**i
        risk_list = []
        r_qbits = i
        print(f"rank {i+1}/{config['x_qbits'] + 1} (rank={rank})")
        for num_points in range(1, config['num_points'] + 1):
            print(f"num_points {num_points}/{config['num_points']}")
            risk = avg_risk(rank, num_points, config['x_qbits'],
                            r_qbits, config['num_unitaries'],
                            config['num_layers'], config['num_training_data'],
                            config['num_epochs']
                            )
            writer.append(risk)
            risk_list.append(risk)
        all_risks.append(risk_list)
    all_risks_array = np.array(all_risks)

    # store risks
    np.save(save_dir + 'cobyla_result.npy', all_risks_array)


def two_points_rank_one():
    risk = avg_risk(1, 2, 1, 1, 10, 10, 10, 120)
    print(risk)


if __name__ == '__main__':
    main()
    # two_points_rank_one()
    # def plot_results(result, save_dir):
    #     import matplotlib.pyplot as plt
    #     plt.plot([1, 2], result[0, :], label='r=1')
    #     plt.plot([1, 2], result[1, :], label='r=2')
    #     plt.legend()
    #     plt.savefig(save_dir)
    #     plt.cla()
    
    # adam_result = np.load('./experimental_results/exp1/result.npy')
    # plot_results(adam_result, './experimental_results/adam_result.png')
    # cobyla_result = np.array([[0.33881214585908626, 0.13059633551205363], [0.11080463275009704, 0.027872138911887306]])
    # cobyla_result = np.load('./experimental_results/exp1/cobyla_result.npy')
    # plot_results(cobyla_result, './experimental_results/cobyla_result1.png')
