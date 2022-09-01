import time

import pennylane as qml
import numpy as np
import torch
from data import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import List
from utils import *
import matplotlib.pyplot as plt


class QNN:

    def __init__(self, wires: List[int], num_layers: int, use_torch=True, device='cpu'):
        self.wires = wires
        self.num_layers = num_layers
        self.use_torch = use_torch
        self.device = device
        self.params = self.init_params()

    @abstractmethod
    def init_params(self):
        """
        Initialises the parameters of the quantum neural network
        """

    @abstractmethod
    def qnn(self):
        """
        Creates qnn circuit on self.wires with self.num_layers many layers
        """


    def get_matrix_V(self):
        if self.use_torch:
            return qml.matrix(self.qnn)().detach().numpy()
        else:
            return qml.matrix(self.qnn)()

    def get_tensor_V(self):
        return qml.matrix(self.qnn)()


class BarrenQNN(QNN):

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
        super(BarrenQNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        # 3 Parameters per qbit per layer, since we have a parameterised X, Y, Z rotation
        if self.use_torch:
            params = np.random.normal(0, 2*np.pi, size = len(self.wires))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, 2*np.pi, size = len(self.wires))

    def entanglement(self):
        if len(self.wires) > 1:
            for i in range(len(self.wires)-1):
                qml.CZ(wires=[i, i + 1])
    def layer(self, layer_num):
        gate_set = [qml.RX, qml.RY, qml.RZ]
        for i in range(len(self.wires)):
            qml.RY(np.pi / 4, wires=i)
        for i in range(len(self.wires)):
            gate = np.random.choice(gate_set)
            gate(self.params[i], wires=i)


        self.entanglement()

    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)











def cost_func(X_train, qnn: QNN, unitary, ref_wires: List[int], dev: qml.Device):
    # input params: train data, qnn, unitary to learn, refernce system wires and device
    cost = torch.zeros(1)
    print('shape of train', np.shape(X_train))
    for el in X_train:
        print(el)
        print(np.shape(el))
        #print(el.shape)
        @qml.qnode(dev, interface="torch")
        def circuit():
            qml.QubitStateVector(el, wires=qnn.wires + ref_wires)  # Amplitude Encoding
            qnn.qnn()
            adjoint_unitary_circuit(unitary)(wires=qnn.wires)  # Adjoint U
            qml.MottonenStatePreparation(el, wires=qnn.wires + ref_wires).inv()  # Inverse Amplitude Encoding
            return qml.probs(wires=qnn.wires + ref_wires)

        cost += circuit()[0]

    return 1 - (cost / len(X_train))



def train_qnn(qnn: QNN, unitary, X_train, ref_wires: List[int],
              dev: qml.Device, learning_rate: int, num_epochs: int):
    # num_qubits = len(qnn.wires) + len(ref_wires)
    # num_layers = qnn.num_layers
    # set up the optimizer
    opt = torch.optim.Adam([qnn.params], lr=learning_rate)
    # opt = torch.optim.SGD([qnn.params], lr=learning_rate)

    # number of steps in the optimization routine
    steps = 1

    # the final stage of optimization isn't always the best, so we keep track of
    # the best parameters along the way
    # best_cost = 0
    # best_params = np.zeros((num_qubits, num_layers, 3))

    # optimization begins
    all_losses = []
    grad_storage = []
    for n in range(steps):
        print(f"step {n + 1}/{steps}")
        opt.zero_grad()
        total_loss = 0
        #for X in X_train:
        print('calc cost funktion')
        print('data of one step',X_train)
        loss = cost_func(X_train, qnn, unitary, ref_wires, dev)
        print('backprop')
        loss.backward()

        print(np.array(qnn.params.grad))
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

    print(all_losses)
    # return all_losses
    #print(grad_storage.detach().cpu().numpy())
    return np.array(qnn.params.grad)


class SchmidtDataset(torch.utils.data.Dataset):

    def __init__(self, schmidt_rank, num_points, x_qbits, r_qbits):
        # Initialize the data and label list
        self.data = uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)
        self.labels = [0] * num_points

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
        self.labels = [0] * num_points

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.data)


def main():
    from quantum_backends import QuantumBackends
    num_samples = 100
    gradient_samples = []
    for i in range(num_samples):
        x_qbits = 3
        schmidt_rank = 1
        num_points = 32
        r_qbits = int(np.ceil(np.log2(schmidt_rank)))
        ref_wires = list(range(x_qbits, x_qbits + r_qbits))
        X_train = np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits))
        dataloader = SchmidtDataset(schmidt_rank, num_points, x_qbits, r_qbits)
        print(dataloader.__getitem__(0))
        dev = qml.device("default.qubit", wires=x_qbits+ len(ref_wires))
        qnn = BarrenQNN(list(range(x_qbits)), 1, use_torch=True)

        unitary= random_unitary_matrix(x_qbits)

        gradient = train_qnn(qnn, unitary,X_train,ref_wires,dev,0.1, 2)
        print('Final gradient', gradient)
        gradient_samples.append(gradient)
    #print(fast_cost_func(X_train, qnn, ref_wires, dev, transpiled_unitary))

    print("Variance of the gradients for {} random circuits: {}".format(
        num_samples, np.var(np.array(gradient_samples),axis= 0)
    )
    )
    print("Mean of the gradients for {} random circuits: {}".format(
        num_samples, np.mean(np.array(gradient_samples), axis = 0)
    )
    )

    qubits = [2, 3, 4, 5, 6]
    variances = []

    for num_qubits in qubits:
        gradient_vals = []
        for i in range(num_samples):
            x_qbits = num_qubits
            schmidt_rank = 1
            num_points = 32
            r_qbits = int(np.ceil(np.log2(schmidt_rank)))
            ref_wires = list(range(x_qbits, x_qbits + r_qbits))
            X_train = np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits))
            dataloader = SchmidtDataset(schmidt_rank, num_points, x_qbits, r_qbits)
            print(dataloader.__getitem__(0))
            dev = qml.device("default.qubit", wires=x_qbits + len(ref_wires))
            qnn = BarrenQNN(list(range(x_qbits)), 1, use_torch=True)

            unitary = random_unitary_matrix(x_qbits)

            gradient = train_qnn(qnn, unitary, X_train, ref_wires, dev, 0.1, 2)
            print('Final gradient', gradient)
            gradient_vals.append(np.linalg.norm(gradient))

        print(gradient_samples)
        variances.append(np.var(gradient_vals))  #np.mean(np.var(, axis=0)), wenn kein np.linalg.norm()

    variances = np.array(variances)
    qubits = np.array(qubits)

    # Fit the semilog plot to a straight line
    p = np.polyfit(qubits, np.log(variances), 1)

    # Plot the straight line fit to the semilog
    plt.semilogy(qubits, variances, "o")
    plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.", label="Slope {:3.2f}".format(p[0]))
    plt.xlabel(r"N Qubits")
    plt.ylabel(r"Variance")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()