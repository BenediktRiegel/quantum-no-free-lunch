from qnn import QNN, PennylaneQNN, OffsetQNN
from utils import uniform_random_data, random_unitary_matrix, torch_tensor_product
import time
import numpy as np
from utils import quantum_risk
import torch
import matplotlib.pyplot as plt

torch.manual_seed(4241)
np.random.seed(4241)


def plot_loss(losses, num_qbits, num_layers):
    plt.plot(list(range(len(losses))), losses)
    plt.title(f"Loss for net with {num_qbits} qbits and {num_layers} layers")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(f"./plots/loss_{num_qbits}_qbits_{num_layers}_layers.png")
    plt.cla()


def cost_func(X, qnn, unitary, r_I):
    cost = torch.zeros((1,))
    V = torch_tensor_product(qnn.get_tensor_V(), r_I)
    for el in X:
        state = torch.matmul(V, el)
        state = torch.matmul(unitary, state)
        state = torch.dot(torch.conj(el), state)
        cost += torch.square(state.real) + torch.square(state.imag)
        # cost += torch.square(torch.abs(torch.dot(el, state)))
    cost /= len(X)
    return 1 - cost


def train(X, qnn, unitary, num_epochs, optimizer, r_I):
    losses = []
    for i in range(num_epochs):
        loss = cost_func(X, qnn, unitary, r_I)
        losses.append(loss.item())
        # if i % 20 == 0:
        #     print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}")
        if loss.item() == 0.0:
            # print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}\nstopped")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch [{i+1}/{num_epochs}] final loss {losses[-1]}")
    return losses


def I(size):
    matrix = torch.zeros((size, size), dtype=torch.complex128)
    for i in range(size):
        matrix[i, i] = 1
    return matrix


def init(num_layers, num_qbits):
    x_qbits = num_qbits
    r_qbits = num_qbits
    x_wires = list(range(num_qbits))
    qnn = PennylaneQNN(wires=x_wires, num_layers=num_layers, use_torch=True)

    print('prep')
    X = torch.from_numpy(np.array(uniform_random_data(2**x_qbits, 2, x_qbits, r_qbits)))
    U = random_unitary_matrix(x_qbits)
    # U = qnn.get_matrix_V()
    U_inv = U.conj().T
    r_I = I(2**r_qbits)
    U_inv_I = torch_tensor_product(torch.from_numpy(U_inv), r_I)

    optimizer = torch.optim.Adam([qnn.params], lr=0.1)

    starting_time = time.time()
    losses = train(X, qnn, U_inv_I, 75, optimizer, r_I)
    total_time = time.time() - starting_time
    plot_loss(losses, num_qbits, num_layers)

    print(f"risk = {quantum_risk(U, qnn.get_matrix_V())}")
    return total_time, losses[-1]


def main():
    # time = init(num_layers=1, num_qbits=1)
    # print(f"{time}s")
    num_layers = [1] + list(range(5, 100, 5))
    qbits = [4]
    final_losses = []
    times = []
    for qbit in qbits:
        for num_layer in num_layers:
            training_time, final_loss = init(num_layer, qbit)
            print(f"Training with {qbit} qubits and {num_layer} layers took {training_time}s")
            final_losses.append(final_loss)
            times.append(training_time)
    plt.plot(num_layers, final_losses)
    plt.xlabel('number of layers')
    plt.ylabel('final loss')
    plt.title('Final loss after 75 epochs for a 4 qubit system')
    plt.savefig('./plots/final_loss.png')
    plt.cla()
    plt.plot(num_layers, times)
    plt.xlabel('number of layers')
    plt.ylabel('runtime [s]')
    plt.title('Run time of 75 epochs for a 4 qubit system')
    plt.savefig('./plots/runtime.png')
    plt.cla()


if __name__ == '__main__':
    main()
