import time

import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import List
from utils import *



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

    @abstractmethod
    def parameter_shift_qnn(self, shift_idx, shift_version):
        """
        Creates qnn circuit, but parameter shifted
        """

    def get_matrix_V(self):
        if self.use_torch:
            return qml.matrix(self.qnn)().detach().numpy()
        else:
            return qml.matrix(self.qnn)()
    
    def get_tensor_V(self):
        return qml.matrix(self.qnn)()


class PennylaneQNN(QNN):

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
        super(PennylaneQNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        # 3 Parameters per qbit per layer, since we have a parameterised X, Y, Z rotation
        if self.use_torch:
            params = np.random.normal(0, np.pi, (len(self.wires), self.num_layers, 3))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, np.pi, (len(self.wires), self.num_layers, 3))

    def entanglement(self):
        if len(self.wires) > 1:
            for i in range(len(self.wires)):
                c_wire = self.wires[i]
                t_i = i % len(self.wires)
                t_wire = self.wires[t_i]
                qml.CNOT(wires=[c_wire, t_wire])

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.RX(self.params[i, layer_num, 0], wires=self.wires[i])
            qml.RY(self.params[i, layer_num, 1], wires=self.wires[i])
            qml.RZ(self.params[i, layer_num, 2], wires=self.wires[i])

        self.entanglement()

    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)

    def get_param_indices(self):
        for wire in range(len(self.wires)):
            for layer in range(self.num_layers):
                for param in range(3):
                    yield (wire, layer, param)

    def parameter_shift_layer(self, layer_num, shifted_wire, shifted_param, shift_factor):
        for i in range(shifted_wire):
            qml.RX(self.params[i, layer_num, 0], wires=self.wires[i])
            qml.RY(self.params[i, layer_num, 1], wires=self.wires[i])
            qml.RZ(self.params[i, layer_num, 2], wires=self.wires[i])

        shifted_params = self.params[shifted_wire, layer_num, :]
        shifted_params[shifted_param] = self.params[shifted_wire, layer_num, shifted_param] + shift_factor*(np.pi / 4.)
        qml.RX(shifted_params[0], wires=shifted_wire)
        qml.RY(shifted_params[1], wires=shifted_wire)
        qml.RZ(shifted_params[2], wires=shifted_wire)

        for i in range(shifted_wire + 1, len(self.wires)):
            qml.RX(self.params[i, layer_num, 0], wires=self.wires[i])
            qml.RY(self.params[i, layer_num, 1], wires=self.wires[i])
            qml.RZ(self.params[i, layer_num, 2], wires=self.wires[i])

        self.entanglement()

    def parameter_shift_qnn(self, shift_idx, shift_factor):
        (shifted_wire, shifted_layer, shifted_param) = shift_idx
        for j in range(shifted_layer):
            self.layer(j)
        self.parameter_shift_layer(shifted_layer, shifted_wire, shifted_param, shift_factor)
        for j in range(shifted_layer+1, self.num_layers):
            self.layer(j)


class OffsetQNN(QNN):

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
        super(OffsetQNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        # 3 Parameters per qbit per layer, since we have a parameterised X, Y, Z rotation
        if self.use_torch:
            params = np.random.normal(0, np.pi, (len(self.wires), self.num_layers, 3))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, np.pi, (len(self.wires), self.num_layers, 3))

    def entanglement(self, layer_num):
        layer_num = layer_num % len(self.wires)
        if layer_num == 0:
            layer_num += 1
        for i in range(len(self.wires)):
            c_wire = self.wires[i]
            t_i = (i+layer_num) % len(self.wires)
            t_wire = self.wires[t_i]
            qml.CNOT(wires=[c_wire, t_wire])

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.RX(self.params[i, layer_num, 0], wires=self.wires[i])
            qml.RY(self.params[i, layer_num, 1], wires=self.wires[i])
            qml.RZ(self.params[i, layer_num, 2], wires=self.wires[i])

        self.entanglement(layer_num)

    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)


def get_density_matrix(qstate):
    qstate = np.array(qstate)
    return np.outer(qstate, qstate.conj())


def cost_func(X_train, qnn: QNN, unitary, ref_wires: List[int], dev: qml.Device):
    #input params: train data, qnn, unitary to learn, refernce system wires and device
    cost = torch.zeros(1)
    for el in X_train:
        @qml.qnode(dev, interface="torch")
        def circuit():
            qml.QubitStateVector(el, wires=qnn.wires+ref_wires)  # Amplitude Encoding
            qnn.qnn()
            adjoint_unitary_circuit(unitary)(wires=qnn.wires)  # Adjoint U
            qml.MottonenStatePreparation(el, wires=qnn.wires+ref_wires).inv()  # Inverse Amplitude Encoding
            return qml.probs(wires=qnn.wires+ref_wires)
        cost += circuit()[0]

    return 1 - (cost / len(X_train))


def fast_cost_func(X_train, qnn: QNN, ref_wires: List[int], dev: qml.Device, transpiled_unitary):
    #input params: train data, qnn, unitary to learn, refernce system wires and device
    cost = torch.zeros(1)
    for el in X_train:
        @qml.qnode(dev)
        def circuit():
            qml.QubitStateVector(el, wires=qnn.wires+ref_wires)  # Amplitude Encoding
            qnn.qnn()
            transpiled_unitary(wires=qnn.wires)  # Adjoint U
            return qml.probs(wires=[0])
        # print('run circuit')
        circuit()
        # print('post processing')
        state = dev._state
        # print(state.shape)
        state = np.reshape(state, (2**(len(state.shape)),))
        prob = abs2(np.inner(el, state))
        cost += prob

    return 1 - (cost / len(X_train))


def train_qnn(qnn: QNN, unitary, dataloader: DataLoader, ref_wires: List[int],
              dev: qml.Device, learning_rate: int, num_epochs: int):
    # num_qubits = len(qnn.wires) + len(ref_wires)
    # num_layers = qnn.num_layers
    # set up the optimizer
    opt = torch.optim.Adam([qnn.params], lr=learning_rate)
    # opt = torch.optim.SGD([qnn.params], lr=learning_rate)

    # number of steps in the optimization routine
    steps = num_epochs

    # the final stage of optimization isn't always the best, so we keep track of
    # the best parameters along the way
    # best_cost = 0
    # best_params = np.zeros((num_qubits, num_layers, 3))

    # optimization begins
    all_losses = []
    for n in range(steps):
        print(f"step {n+1}/{steps}")
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

    print(all_losses)
    # return all_losses


class SchmidtDataset(torch.utils.data.Dataset):
    from utils import uniform_random_data
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
    from utils import uniform_random_data
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

def main():
    from quantum_backends import QuantumBackends
    x_qbits = 1
    X_train = np.array([[0, 0, 0, 0]])
    dev = QuantumBackends.qml_lightning.get_pennylane_backend('', '', x_qbits*2)
    qnn = PennylaneQNN(list(range(x_qbits)), 1, use_torch=False)
    ref_wires = list(range(x_qbits, 2*x_qbits))
    unitary = np.array([
        [0, 1],
        [1, 0]
    ])
    qnn.params = np.array([
        [
            [np.pi*0.75, 0, 0]
        ]
    ])
    transpiled_unitary = adjoint_unitary_circuit(unitary)
    print(fast_cost_func(X_train, qnn, ref_wires, dev, transpiled_unitary))


if __name__ == '__main__':
    main()
