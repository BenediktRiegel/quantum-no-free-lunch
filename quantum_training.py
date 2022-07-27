import torch
import pennylane as qml
from qnns.qnn import QNN
from typing import List
from data import adjoint_unitary_circuit
from torch.utils.data import DataLoader


def cost_func(X_train, qnn: QNN, unitary, ref_wires: List[int], dev: qml.Device):
    """
    X_train: 2D torch.tensor containing the training data
    qnn: quantum neural network
    unitary: is the unitary U that needs to be learned
    ref_wires: qubits of the reference system
    dev: quantum device to execute quantum circuits

    Method computes the cost function 1 - (prob_0 / |X_train|)
    where prob_0 is the sum of all the circuits' probability be in state |00>.
    For each training vector in X_train we compute the following circuit's
    probability to end up in state |00>

    Circuit:
    A^* U^* V A |00>
    """
    #input params: train data, qnn, unitary to learn, refernce system wires and device
    cost = torch.zeros(1)
    for el in X_train:
        @qml.qnode(dev, interface="torch")
        def circuit():
            qml.MottonenStatePreparation(el, wires=qnn.wires+ref_wires)         # Amplitude Encoding
            qnn.qnn()                                                           # V (qnn)
            adjoint_unitary_circuit(unitary)(wires=qnn.wires)                   # Adjoint U
            qml.MottonenStatePreparation(el, wires=qnn.wires+ref_wires).inv()   # Inverse Amplitude Encoding
            return qml.probs(wires=qnn.wires+ref_wires)                         # Return probabilities for differen quantum states
        cost += circuit()[0]                                                    # Sum up probability of state |0>

    return 1 - (cost / len(X_train))                                            # Final cost function


def train(qnn: QNN, unitary, dataloader: DataLoader, ref_wires: List[int],
              dev: qml.Device, learning_rate: int, num_epochs: int):
    """
    qnn: quantum neural network (V)
    unitary: unitary U that qnn should learn
    dataloader: a dataloader that contains all the training data
    learning_rate: the learning rate
    num_epochs: the number of epochs

    The train method currently uses pytorch's optimizers for the backpropagation, in order to train the quantum neural network (qnn)
    """
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