"""
This file contains all qnn architectures
"""

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
        depth = len(self.wires) + 3
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi
        # 3 Parameters per qbit per layer, since we have a parameterised X, Y, Z rotation

        if self.use_torch:
            params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 3))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 3))

    def entanglement(self):
        if len(self.wires) > 1:
            for i in range(len(self.wires)):
                c_wire = self.wires[i]
                t_i = (i+1) % len(self.wires)
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
        depth = len(self.wires) + 3
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi

        if self.use_torch:
            params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 3))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 3))

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

class Circuit2QNN(QNN):
    """
    Paper Expressibility of QNNs
    Circuit 2
    """


    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
        super(Circuit2QNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        depth = len(self.wires) + 1
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi

        if self.use_torch:
            params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 2))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 2))

    def entanglement(self):
        if len(self.wires) > 1:
            for i in range(len(self.wires)-1, 0, -1):
                c_wire = self.wires[i]
                t_i = i-1
                t_wire = self.wires[t_i]
                qml.CNOT(wires=[c_wire, t_wire])

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.RX(self.params[i, layer_num, 0], wires=self.wires[i])
            qml.RZ(self.params[i, layer_num, 1], wires=self.wires[i])

        self.entanglement()

    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)

class Circuit5QNN(QNN):
    """
    Paper Expressibility of QNNs
    Circuit 5
    """

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
        super(Circuit5QNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        # wall_params = |wires| x |layers| x 4
        # cnot_params = |layers| x |wires|*(|wires|-1)
        depth = len(self.wires) * (len(self.wires)-1) + 4
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi

        if self.use_torch:
            wall_params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 4))
            cnot_params = np.random.normal(0, std_dev, (self.num_layers, len(self.wires)*(len(self.wires)-1)))
            wall_params = Variable(torch.tensor(wall_params), requires_grad=True)
            cnot_params = Variable(torch.tensor(cnot_params), requires_grad=True)
        else:
            wall_params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 4))
            cnot_params = np.random.normal(0, std_dev, (self.num_layers, len(self.wires) * (len(self.wires) - 1)))
        return [wall_params, cnot_params]

    def entanglement(self, num_layer):
        idx = 0
        for c_wire in self.wires:
            for t_wire in self.wires:
                if c_wire != t_wire:
                    qml.CRZ(self.params[1][num_layer][idx], wires=(c_wire, t_wire))
                    idx += 1

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.RX(self.params[0][i, layer_num, 0], wires=self.wires[i])
            qml.RZ(self.params[0][i, layer_num, 1], wires=self.wires[i])

        self.entanglement(layer_num)

        for i in range(len(self.wires)):
            qml.RX(self.params[0][i, layer_num, 2], wires=self.wires[i])
            qml.RZ(self.params[0][i, layer_num, 3], wires=self.wires[i])

    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)


class Circuit6QNN(QNN):
    """
    Paper Expressiblity of QNNs
    Circuit 6
    """

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
        super(Circuit6QNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        # wall_params = |wires| x |layers| x 4
        # cnot_params = |layers| x |wires|*(|wires|-1)
        depth = len(self.wires) * (len(self.wires)-1) + 4
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi

        if self.use_torch:
            wall_params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 4))
            cnot_params = np.random.normal(0, std_dev, (self.num_layers, len(self.wires)*(len(self.wires)-1)))
            wall_params = Variable(torch.tensor(wall_params), requires_grad=True)
            cnot_params = Variable(torch.tensor(cnot_params), requires_grad=True)
        else:
            wall_params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 4))
            cnot_params = np.random.normal(0, std_dev, (self.num_layers, len(self.wires) * (len(self.wires) - 1)))
        return [wall_params, cnot_params]

    def entanglement(self, num_layer):
        idx = 0
        for c_wire in self.wires:
            for t_wire in self.wires:
                if c_wire != t_wire:
                    qml.CRX(self.params[1][num_layer][idx], wires=(c_wire, t_wire))
                    idx += 1

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.RX(self.params[0][i, layer_num, 0], wires=self.wires[i])
            qml.RZ(self.params[0][i, layer_num, 1], wires=self.wires[i])

        self.entanglement(layer_num)

        for i in range(len(self.wires)):
            qml.RX(self.params[0][i, layer_num, 2], wires=self.wires[i])
            qml.RZ(self.params[0][i, layer_num, 3], wires=self.wires[i])


    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)

class Circuit9QNN(QNN):
    """
    Paper Expressibility of QNN 
    Circuit 9
    """

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
            super(Circuit9QNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        depth = len(self.wires)
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi

        if self.use_torch:
            params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 1))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 1))

    def entanglement(self):
        if len(self.wires) > 1:
            for i in range(len(self.wires)-1, 0, -1):
                c_wire = self.wires[i]
                t_i = i-1
                t_wire = self.wires[t_i]
                qml.CZ(wires=[c_wire, t_wire])

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.Hadamard(wires=self.wires[i])

        self.entanglement()

        for i in range(len(self.wires)):
            qml.RX(self.params[i, layer_num, 0], wires=self.wires[i])


    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)


class Circuit11QNN(QNN):
    """
    Paper Expressibility of QNN
    Circuit 11
    """

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
            super(Circuit11QNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        depth = 6 * self.num_layers
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi

        if self.use_torch:
            outer_params = np.random.normal(0, std_dev, (2, self.num_layers, 2))
            inner_params = np.random.normal(0, std_dev, (len(self.wires)-2, self.num_layers, 4))
            outer_params = Variable(torch.tensor(outer_params), requires_grad=True)
            inner_params = Variable(torch.tensor(inner_params), requires_grad=True)
        else:
            outer_params = np.random.normal(0, std_dev, (2, self.num_layers, 2))
            inner_params = np.random.normal(0, std_dev, (len(self.wires) - 2, self.num_layers, 4))
        return [outer_params, inner_params]

    def entanglement_big(self):
        for i in range(0, len(self.wires),2):
            t_wire = self.wires[i]
            c_wire = self.wires[i+1]
            qml.CNOT(wires=[c_wire, t_wire])

    def entanglement_small(self):
        for i in range(1,len(self.wires)-1, 2):
            t_wire = self.wires[i]
            c_wire = self.wires[i + 1]
            qml.CNOT(wires=[c_wire, t_wire])


    def layer(self, layer_num):
        qml.RY(self.params[0][0, layer_num, 0], wires=self.wires[0])
        qml.RZ(self.params[0][0, layer_num, 1], wires=self.wires[0])

        qml.RY(self.params[0][1, layer_num, 0], wires=self.wires[-1])
        qml.RZ(self.params[0][1, layer_num, 1], wires=self.wires[-1])

        for i in range(1,len(self.wires)-1):
            qml.RY(self.params[1][i-1, layer_num, 0],wires=self.wires[i])
            qml.RZ(self.params[1][i-1, layer_num, 1],wires=self.wires[i])

        self.entanglement_big()

        for i in range(1,len(self.wires)-1):
            qml.RY(self.params[1][i - 1, layer_num, 2], wires=self.wires[i])
            qml.RZ(self.params[1][i - 1, layer_num, 3], wires=self.wires[i])
        self.entanglement_small()


    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)

class Circuit12QNN(QNN):
    """
    Paper Expressibility of QNN
    Circuit 12
    """

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
            super(Circuit12QNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        depth = 6 * self.num_layers
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi
     
        if self.use_torch:
            outer_params = np.random.normal(0, std_dev, (2, self.num_layers, 2))
            inner_params = np.random.normal(0, std_dev, (len(self.wires)-2, self.num_layers,4))
            outer_params = Variable(torch.tensor(outer_params), requires_grad=True)
            inner_params = Variable(torch.tensor(inner_params), requires_grad=True)
        else:
            outer_params = np.random.normal(0, std_dev, (2, self.num_layers, 2))
            inner_params = np.random.normal(0, std_dev, (len(self.wires) - 2, self.num_layers, 4))
        return [outer_params, inner_params]

    def entanglement_big(self):
        for i in range(0, len(self.wires),2):
            c_wire = self.wires[i]
            t_wire = self.wires[i+1]
            qml.CZ(wires=[c_wire, t_wire])

    def entanglement_small(self):
        for i in range(1,len(self.wires)-1, 2):
            c_wire = self.wires[i]
            t_wire = self.wires[i + 1]
            qml.CZ(wires=[c_wire, t_wire])


    def layer(self, layer_num):
        qml.RY(self.params[0][0, layer_num, 0], wires=self.wires[0])
        qml.RZ(self.params[0][0, layer_num, 1], wires=self.wires[0])

        qml.RY(self.params[0][1, layer_num, 0], wires=self.wires[-1])
        qml.RZ(self.params[0][1, layer_num, 1], wires=self.wires[-1])

        for i in range(1,len(self.wires)-1):
            qml.RY(self.params[1][i-1, layer_num, 0],wires=self.wires[i])
            qml.RZ(self.params[1][i-1, layer_num, 1],wires= self.wires[i])

        self.entanglement_big()

        for i in range(1,len(self.wires)-1):
            qml.RY(self.params[1][i - 1, layer_num, 2], wires=self.wires[i])
            qml.RZ(self.params[1][i - 1, layer_num, 3], wires=self.wires[i])
        self.entanglement_small()


    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)


class Circuit13QNN(QNN):
    """
    Paper Expressibility of QNN
    Circuit 13
    """

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
            super(Circuit13QNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        depth = 2 * len(self.wires) + 2
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi
     
        if self.use_torch:
            params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 4))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 4))

    def entanglement_1(self, layer_num):
        c_wire = self.wires[-1]
        t_wire = self.wires[0]
        qml.CRZ(self.params[c_wire, layer_num, 1], wires=[c_wire, t_wire])
        if len(self.wires) > 1:
            for i in range(0, len(self.wires)-1):
                c_wire = self.wires[i]
                t_wire = self.wires[i+1]
                qml.CRZ(self.params[c_wire, layer_num, 1], wires=[c_wire, t_wire])

    def entanglement_2(self, layer_num):
        c_wire = self.wires[-1]
        t_wire = self.wires[-2]
        qml.CRZ(self.params[c_wire, layer_num, 3], wires=[c_wire, t_wire])

        c_wire = self.wires[0]
        t_wire = self.wires[-1]
        qml.CRZ(self.params[c_wire, layer_num, 3], wires=[c_wire, t_wire])
        if len(self.wires) > 1:
            for i in range(1, len(self.wires)-1):
                c_wire = self.wires[i-1]
                t_wire = self.wires[i]
                qml.CRZ(self.params[c_wire, layer_num, 3], wires=[c_wire, t_wire])

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.RY(self.params[i, layer_num, 0],wires=self.wires[i])

        self.entanglement_1(layer_num)

        for i in range(len(self.wires)):
            qml.RY(self.params[i, layer_num, 2], wires=self.wires[i])

        self.entanglement_2(layer_num)


    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)


class Circuit14QNN(QNN):
    """
    Paper Expressibility of QNN
    Circuit 14
    """

    def __init__(self, wires: List[int], num_layers: int, use_torch=False, device='cpu'):
            super(Circuit14QNN, self).__init__(wires, num_layers, use_torch, device)

    def init_params(self):
        depth = 2 * len(self.wires) + 2
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi
     
        if self.use_torch:
            params = np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 4))
            return Variable(torch.tensor(params), requires_grad=True)
        else:
            return np.random.normal(0, std_dev, (len(self.wires), self.num_layers, 4))

    def entanglement_1(self, layer_num):
        c_wire = self.wires[-1]
        t_wire = self.wires[0]
        qml.CRX(self.params[c_wire, layer_num, 1], wires=[c_wire, t_wire])
        if len(self.wires) > 1:
            for i in range(0, len(self.wires)-1):
                c_wire = self.wires[i]
                t_wire = self.wires[i+1]
                qml.CRX(self.params[c_wire, layer_num, 1], wires=[c_wire, t_wire])

    def entanglement_2(self, layer_num):
        c_wire = self.wires[-1]
        t_wire = self.wires[-2]
        qml.CRZ(self.params[c_wire, layer_num, 3], wires=[c_wire, t_wire])

        c_wire = self.wires[0]
        t_wire = self.wires[-1]
        qml.CRX(self.params[c_wire, layer_num, 3], wires=[c_wire, t_wire])
        if len(self.wires) > 1:
            for i in range(1, len(self.wires)-1):
                c_wire = self.wires[i-1]
                t_wire = self.wires[i]
                qml.CRX(self.params[c_wire, layer_num, 3], wires=[c_wire, t_wire])

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.RY(self.params[i, layer_num, 0],wires=self.wires[i])

        self.entanglement_1(layer_num)

        for i in range(len(self.wires)):
            qml.RY(self.params[i, layer_num, 2], wires=self.wires[i])

        self.entanglement_2(layer_num)


    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)

