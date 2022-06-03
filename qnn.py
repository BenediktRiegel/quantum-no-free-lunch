import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from abc import abstractmethod
from typing import List



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


def train_net(dataset: Dataset):
    pass


def run_qnn():
    wires = list(range(5))
    num_layers = 3
    dev = qml.device('default.qubit', wires=wires)
    p_qnn = PennylaneQNN(wires, num_layers)

    @qml.device(dev)
    def circuit():
        p_qnn.qnn()
        return qml.probs(wires=wires)

    return float(circuit()[0])

def zero_prob(circuit, params: Variable, wires):
    @qml.qnode(dev)
    def measure_z():
        circuit(params)
        return qml.probs(wires=wires)

    return measure_z()[0]

# cost function
def cost_fn(params):
    cost = 0
    for k in range(3):
        cost += torch.abs(circuit(params, Paulis[k]) - bloch_v[k])

    return cost


# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of steps in the optimization routine
steps = 200

# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(params)
best_params = np.zeros((nr_qubits, nr_layers, 3))

print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))

# optimization begins
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params)
    loss.backward()
    opt.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
        best_params = params

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# calculate the Bloch vector of the output state
output_bloch_v = np.zeros(3)
for l in range(3):
    output_bloch_v[l] = circuit(best_params, Paulis[l])

# print results
print("Target Bloch vector = ", bloch_v.numpy())
print("Output Bloch vector = ", output_bloch_v)