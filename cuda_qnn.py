import torch
from abc import abstractmethod
import numpy as np
from torch.autograd import Variable
import pennylane as qml
import quantum_gates as qg


class CudaQNN:

    def __init__(self, num_wires, num_layers: int, device='cpu'):
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
        return self.qnn().detach().numpy()

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
            for _ in range(self.num_layers):
                ent_layers.append(torch.tensor(qml.matrix(ent_layer)(), device=self.device, dtype=torch.complex128))
            return ent_layers

    def init_params(self):
        depth = self.num_wires + 3
        depth *= self.num_layers
        std_dev = np.sqrt(1/depth)
        # std_dev = np.pi
        # 3 Parameters per qbit per layer, since we have a parameterised X, Y, Z rotation

        params = np.random.normal(0, std_dev, (self.num_wires, self.num_layers, 3))
        return Variable(torch.tensor(params), requires_grad=True)

    def layer(self, layer_num):
        result = qg.U3(self.params[0, layer_num, 0], self.params[0, layer_num, 1], self.params[0, layer_num, 2])
        for i in range(1, self.num_wires):
            result = qg.torch_tensor(
                result,
                qg.U3(self.params[i, layer_num, 0], self.params[i, layer_num, 1], self.params[i, layer_num, 2])
            )
        result.to(self.device)
        if self.num_wires > 1:
            result = torch.matmul(self.ent_layers[layer_num], result)
        return result

    def qnn(self):
        result = self.layer(0)
        for j in range(1, self.num_layers):
            result = torch.matmul(self.layer(j), result)
        return result
