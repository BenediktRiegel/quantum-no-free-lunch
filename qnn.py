import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import List
from utils import * #unitary_circuit, adjoint_unitary_circuit, quantum_risk



class QNN:

    def __init__(self, wires: List[int], num_layers: int):
        self.wires = wires
        self.num_layers = num_layers
        self.params = self.init_params()

    @abstractmethod
    def init_params(self) -> Variable:
        """
        Initialises the parameters of the quantum neural network
        """

    @abstractmethod
    def qnn(self):
        """
        Creates qnn circuit on self.wires with self.num_layers many layers
        """

    def get_matrix_V(self):
        return qml.matrix(self.qnn)().detach().numpy()


class PennylaneQNN(QNN):

    def __init__(self, wires: List[int], num_layers: int):
        super(PennylaneQNN, self).__init__(wires, num_layers)

    def init_params(self):
        # 3 Parameters per qbit per layer, since we have a parameterised X, Y, Z rotation
        params = np.random.normal(0, np.pi, (len(self.wires), self.num_layers, 3))
        return Variable(torch.tensor(params), requires_grad=True)

    def layer(self, layer_num):
        for i in range(len(self.wires)):
            qml.RX(self.params[i, layer_num, 0], wires=self.wires[i])
            qml.RY(self.params[i, layer_num, 1], wires=self.wires[i])
            qml.RZ(self.params[i, layer_num, 2], wires=self.wires[i])

        for i in range(len(self.wires) - 1):
            c_wire = self.wires[i]
            t_wire = self.wires[i+1]
            qml.CNOT(wires=[c_wire, t_wire])

    def qnn(self):
        for j in range(self.num_layers):
            self.layer(j)


def cost_func(X_train, qnn: QNN, unitary, ref_wires: List[int], dev: qml.Device):
    #input params: train data, qnn, unitary to learn, refernce system wires and device
    cost = torch.zeros(1)
    for el in X_train:
        @qml.qnode(dev, interface="torch")
        def circuit():
            # print(f"all wires {qnn.wires+ref_wires}")
            # print(f"{len(el)} should equal 2**{len(qnn.wires)+len(ref_wires)} = {2**(len(qnn.wires)+len(ref_wires))}")
            qml.MottonenStatePreparation(el, wires=qnn.wires+ref_wires)  # Amplitude Encoding
            qnn.qnn()
            adjoint_unitary_circuit(unitary)(wires=qnn.wires)  # Adjoint U
            qml.MottonenStatePreparation(el, wires=qnn.wires+ref_wires).inv()  # Inverse Amplitude Encoding
            return qml.probs(wires=qnn.wires+ref_wires)
        cost += circuit()[0]

    return 1 - (cost / len(X_train))




def train_qnn(qnn: QNN, unitary, dataloader: DataLoader, ref_wires: List[int],
              dev: qml.Device, learning_rate: int, num_epochs: int):
    # num_qubits = len(qnn.wires) + len(ref_wires)
    # num_layers = qnn.num_layers
    # set up the optimizer
    opt = torch.optim.Adam([qnn.params], lr=learning_rate)

    # number of steps in the optimization routine
    steps = num_epochs

    # the final stage of optimization isn't always the best, so we keep track of
    # the best parameters along the way
    # best_cost = 0
    # best_params = np.zeros((num_qubits, num_layers, 3))

    # optimization begins
    # all_losses = []
    for n in range(steps):
        print(f"step {n+1}/{steps}")
        opt.zero_grad()
        total_loss = 0
        for X in dataloader:
            loss = cost_func(X[0], qnn, unitary, ref_wires, dev)
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # all_losses.append(total_loss)
        # Keep track of progress every 10 steps
        if n % 10 == 9 or n == steps - 1:
            print(f"Cost after {n + 1} steps is {total_loss}")
        if total_loss == 0.0:
            print(f"loss({total_loss}) = 0.0")
            break

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


def main():
    schmidt_rank = 4
    num_points = 1
    x_qbits = 2
    r_qbits = 2
    qnn_wires = list(range(x_qbits))
    ref_wires = list(range(x_qbits, x_qbits+r_qbits))


    unitary = random_unitary_matrix(x_qbits)
    dataset = SchmidtDataset(schmidt_rank, num_points, x_qbits, r_qbits)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    qnn = PennylaneQNN(wires=qnn_wires, num_layers=10)


    # provider = IBMQ.load_account()
    # dev = qml.device(
    #             "qiskit.ibmq",
    #             wires=qnn_wires + ref_wires,
    #             backend="ibmq_qasm_simulator",
    #             provider=provider,
    #         )
    # dev = qml.device("qiskit.aer", wires=qnn_wires + ref_wires, backend="statevector_simulator")
    dev = qml.device("default.qubit", wires=qnn_wires+ref_wires)
    train_qnn(qnn, unitary, dataloader, ref_wires, dev)
    print("unitary U:")
    print(np.abs(unitary))
    print("qnn unitary V:")
    print(np.abs(qnn.get_matrix_V()))


if __name__ == '__main__':
    main()
