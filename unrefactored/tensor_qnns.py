from qnn import QNN
from typing import List
import torch
import numpy as np
from utils import torch_tensor_product
import pennylane as qml


def RX(th, device='cpu'):
    cos = torch.cos(th/2.)
    sin = torch.sin(th/2.)
    id = I(2, device)
    X = torch.tensor([
        [0, 1],
        [1, 0]
    ], dtype=torch.complex128, device=device).cuda()
    return id*cos - 1j*sin*X


def RY(th, device='cpu'):
    cos = torch.cos(th / 2.)
    sin = torch.sin(th / 2.)
    id = I(2, device)
    tr = torch.tensor([
        [0, 1],
        [0, 0]
    ], dtype=torch.complex128, device=device).cuda()
    bl = torch.tensor([
        [0, 0],
        [1, 0]
    ], dtype=torch.complex128, device=device).cuda()
    return id*cos - tr*sin + bl*sin


def RZ(th, device='cpu'):
    exp = torch.exp(1j*th)
    tl = torch.tensor([
        [1, 0],
        [0, 0]
    ], dtype=torch.complex128, device=device).cuda()
    br = torch.tensor([
        [0, 0],
        [0, 1]
    ], dtype=torch.complex128, device=device).cuda()
    return tl + exp*br


def I(size, device):
    matrix = torch.zeros((size, size), dtype=torch.complex128, device=device)
    for i in range(size):
        matrix[i, i] = 1
    return matrix


class TensorQNN(QNN):

    def __init__(self, wires: List[int], num_layers: int, use_torch=True, device='cpu'):
        super().__init__(wires, num_layers, use_torch, device)
        self.entanglement_layers = self.init_entanglement_layers()
        self.size = 2**len(self.wires)

    def init_params(self):
        # init_params = np.random.normal(0, np.pi, (len(self.wires), self.num_layers, 3))
        # return torch.tensor(init_params, requires_grad=True, device=self.device)
        init_params = np.random.normal(0, np.pi, (len(self.wires)*self.num_layers*3))
        params = []
        for init_p in init_params:
            params.append(torch.tensor(init_p, requires_grad=True, device=self.device))
        return params

    def get_param(self, wire, layer, param_i):
        idx = (wire*self.num_layers*3) + (layer*3) + param_i
        return self.params[idx]
    # def get_param(self, wire, layer, param_i):
    #     return self.params[wire, layer, param_i]

    def entanglement(self):
        if len(self.wires) > 1:
            for i in range(len(self.wires)):
                c_wire = self.wires[i]
                t_i = (i+1) % len(self.wires)
                t_wire = self.wires[t_i]
                qml.CNOT(wires=[c_wire, t_wire])

    def init_entanglement_layers(self):
        if len(self.wires) > 1:
            entanglement_layers = []
            for layer in range(self.num_layers):
                ent_layer = qml.matrix(self.entanglement)()
                entanglement_layers.append(torch.tensor(ent_layer, dtype=torch.complex128, device=self.device).cuda())
            return entanglement_layers
        else:
            return None

    def layer(self, layer_num):
        layer = RX(self.get_param(0, layer_num, 0), self.device)
        layer = torch.matmul(RY(self.get_param(0, layer_num, 1), self.device), layer)
        layer = torch.matmul(RZ(self.get_param(0, layer_num, 2), self.device), layer)
        for i in range(1, len(self.wires)):
            param_part = RX(self.get_param(i, layer_num, 0), self.device)
            param_part = torch.matmul(RY(self.get_param(i, layer_num, 1), self.device), param_part)
            param_part = torch.matmul(RZ(self.get_param(i, layer_num, 2), self.device), param_part)
            layer = torch_tensor_product(layer, param_part, device=self.device)

        if len(self.wires) > 1:
            layer = torch.matmul(self.entanglement_layers[layer_num], layer)

        return layer

    def qnn(self):
        qnn = self.layer(0)
        for i in range(1, self.num_layers):
            qnn = torch.matmul(self.layer(i), qnn)
        return qnn

    def parameter_shift_qnn(self, shift_idx, shift_version):
        raise NotImplementedError("TensorQNN has not implemented parameter_shift_qnn")

    def get_matrix_V(self):
        if self.use_torch:
            return self.qnn().cpu().detach().numpy()
        else:
            return self.qnn().cpu()

    def get_tensor_V(self):
        return self.qnn()


def is_unitary(M, device, error=1e-14):
    diff = torch.matmul(M, M.adjoint()) - I(M.shape[0], device)
    abs_real = torch.abs(diff.real)
    abs_imag = torch.abs(diff.imag)
    return (abs_real < error).all() and (abs_imag < error).all()


def test_qnn():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_qbits = 4
    num_layers = 10
    x_wires = list(range(num_qbits))
    qnn = TensorQNN(wires=x_wires, num_layers=num_layers, use_torch=True, device=device)
    for layer in range(qnn.num_layers):
        for i in range(len(qnn.wires)):
            for param in range(3):
                if param == 0:
                    rotation = RX(qnn.get_param(i, layer, param), device)
                elif param == 1:
                    rotation = RY(qnn.get_param(i, layer, param), device)
                else:
                    rotation = RZ(qnn.get_param(i, layer, param), device)
                if not is_unitary(rotation, device):
                    error_msg = f"rotation matrix qbit {i}, layer {layer}, param {param} is not unitary"
                    raise ValueError(error_msg)
    if qnn.entanglement_layers is not None:
        for layer in range(qnn.num_layers):
            entangle_m = qnn.entanglement_layers[layer]
            if not is_unitary(entangle_m, device):
                error_msg = f"entanglement matrix layer {layer} is not unitary"
                raise ValueError(error_msg)
    print(torch.matmul(qnn.get_tensor_V(), qnn.get_tensor_V().adjoint()))
    if is_unitary(qnn.get_tensor_V(), device):
        print('qnn is a unitary')
    else:
        raise ValueError("qnn is not a unitary")


def main():
    test_qnn()


if __name__ == '__main__':
    main()

