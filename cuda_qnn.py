import time

import torch
from abc import abstractmethod
import numpy as np
import pennylane as qml
import quantum_gates as qg


class CudaQNN:

    def __init__(self, num_wires, num_layers: int, device='cpu'):
        qg.init_globals(device=device)
        self.num_wires = num_wires
        self.num_layers = num_layers
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
        return self.qnn().detach()

    def get_tensor_V(self):
        return self.qnn()


class CudaPennylane(CudaQNN):

    def __init__(self, num_wires, num_layers: int, device='cpu'):
        super(CudaPennylane, self).__init__(num_wires, num_layers, device)
        self.ent_layers = self.init_entanglement_layers()
        self.matrix_size = (2**self.num_wires, 2**self.num_wires)

    def init_entanglement_layers(self):
        if self.num_wires > 1:
            ent_layers = []
            def ent_layer():
                if self.num_wires > 1:
                    for i in range(self.num_wires):
                        c_wire = i
                        t_wire = (i + 1) % self.num_wires
                        qml.CNOT(wires=[c_wire, t_wire])
            return torch.tensor(qml.matrix(ent_layer)(), device=self.device, dtype=torch.complex128)

    def init_params(self):
        depth = self.num_wires + 3
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi
        # 3 Parameters per qbit per layer, since we have a parameterised X, Y, Z rotation

        params = np.random.normal(0, std_dev, (self.num_wires, self.num_layers, 3))
        # return Variable(torch.tensor(params, device=self.device), requires_grad=True)
        return torch.tensor(params, device=self.device, requires_grad=True)

    def layer(self, layer_num):
        result = qg.U3(self.params[0, layer_num, 0], self.params[0, layer_num, 1], self.params[0, layer_num, 2])
        for i in range(1, self.num_wires):
            result = torch.kron(
                result,
                qg.U3(self.params[i, layer_num, 0], self.params[i, layer_num, 1], self.params[i, layer_num, 2])
            )
        result.to(self.device)
        if self.num_wires > 1:
            result = torch.matmul(self.ent_layers, result)
        return result

    def qnn(self):
        result = self.layer(0)
        for j in range(1, self.num_layers):
            result = torch.matmul(self.layer(j), result)
        return result


class CudaCircuit6(CudaQNN):

    def __init__(self, num_wires, num_layers: int, device='cpu'):
        super(CudaCircuit6, self).__init__(num_wires, num_layers, device)
        self.matrix_size = (2**self.num_wires, 2**self.num_wires)
        self.zero = torch.tensor(0)

    def init_params(self):
        depth = self.num_wires * (self.num_wires - 1) + 4
        depth *= self.num_layers
        std_dev = np.sqrt(1 / depth)
        # std_dev = np.pi

        wall_params = np.random.normal(0, std_dev, (self.num_wires, self.num_layers, 4))
        cnot_params = np.random.normal(0, std_dev, (self.num_layers, self.num_wires, self.num_wires - 1))
        wall_params = torch.tensor(wall_params, requires_grad=True)
        cnot_params = torch.tensor(cnot_params, requires_grad=True)
        return [wall_params, cnot_params]

    def ent_up(self, wire, layer_num):
        idx = self.num_wires - wire - 1
        result = qg.controlled_U(wire, wire - 1, qg.RX(self.params[1][layer_num, wire, idx]))
        for other_wire in range(wire - 2, -1, -1):
            idx += 1
            result = qg.quick_matmulmat(
                qg.controlled_U(wire, other_wire, qg.RX(self.params[1][layer_num, wire, idx])), result)
        return result

    def ent_down(self, wire, layer_num):
        idx = 0
        I = qg.I(2)
        result = qg.controlled_U(wire, wire + 1, qg.RX(self.params[1][layer_num, wire, idx]).T)
        for other_wire in range(wire+2, self.num_wires):
            idx += 1
            result = torch.matmul(qg.controlled_U(wire, other_wire, qg.RX(self.params[1][layer_num, wire, idx])).T, torch.kron(result, I))
        return result.T

    def wire_entanglement(self, wire, layer_num):
        if wire == 0:
            return self.ent_down(wire, layer_num)
        elif wire == self.num_wires-1:
            return self.ent_up(wire, layer_num)
        else:
            up = self.ent_up(wire, layer_num)
            down = self.ent_down(wire, layer_num)
            # First Block IoD, then Block UoI. This translates into UoI * IoD, which can be done by quick_matmulmat
            # How big should I in UoI be? num_wires - wire - 1. Example: num_wires = 4, wire = 2, then there is
            # only one wire (wire=3) left below the chosen one.
            return qg.quick_matmulmat(torch.kron(up, qg.I(2**(self.num_wires - wire - 1))), down)

    def ent_layer(self, layer_num):
        result = self.wire_entanglement(0, layer_num)
        for wire in range(1, self.num_wires):
            result = torch.matmul(result, self.wire_entanglement(wire, layer_num))
        return result

    def layer(self, layer_num):
        result = qg.U3(self.params[0][0, layer_num, 0], self.zero, self.params[0][0, layer_num, 2])
        for i in range(1, self.num_wires):
            result = torch.kron(
                result,
                qg.U3(self.params[0][i, layer_num, 0], self.zero, self.params[0][i, layer_num, 2])
            )

        if self.num_wires > 1:
            result = torch.matmul(self.ent_layer(layer_num), result)

        second_wall = qg.U3(self.params[0][0, layer_num, 2], self.zero, self.params[0][0, layer_num, 3])
        for i in range(1, self.num_wires):
            second_wall = torch.kron(
                second_wall,
                qg.U3(self.params[0][i, layer_num, 2], self.zero, self.params[0][i, layer_num, 3])
            )
        result = torch.matmul(second_wall, result)
        result.to(self.device)
        return result

    def qnn(self):
        result = self.layer(0)
        for j in range(1, self.num_layers):
            result = torch.matmul(self.layer(j), result)
        return result
