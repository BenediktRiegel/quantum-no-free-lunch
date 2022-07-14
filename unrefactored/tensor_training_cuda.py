from qnn import QNN, PennylaneQNN, OffsetQNN
from utils import uniform_random_data, random_unitary_matrix, torch_tensor_product
import time
import numpy as np
from utils import quantum_risk
import torch
import matplotlib.pyplot as plt
from tensor_qnns import TensorQNN
import os

directory = '/'.join(os.path.realpath(__file__).replace('\\', '/').split('/')[:-1])

seed = 4241
torch.manual_seed(seed)
np.random.seed(seed)


def plot_loss(losses, num_qbits, num_layers):
    plt.plot(list(range(len(losses))), losses)
    plt.title(f"Loss for net with {num_qbits} qbits and {num_layers} layers")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(f"{directory}/plots/loss_{num_qbits}_qbits_{num_layers}_layers.png")
    plt.cla()


def cost_func(X, qnn, unitary, r_I, device):
    cost = torch.zeros((1,), device=device).cuda()
    V = torch_tensor_product(qnn.get_tensor_V(), r_I, device=device)
    # print(qnn.params)
    # print(V)
    for el in X:
        state = torch.matmul(V, el)
        state = torch.matmul(unitary, state)
        state = torch.dot(torch.conj(el), state)
        cost += (torch.square(state.real) + torch.square(state.imag))
        # cost += torch.square(torch.abs(torch.dot(el, state)))
    cost /= len(X)
    return 1 - cost


def train(X, qnn, unitary, num_epochs, optimizer, r_I, device):
    losses = []
    for i in range(num_epochs):
        loss = cost_func(X, qnn, unitary, r_I, device)
        losses.append(loss.item())
        # if i % 20 == 0:
        #     print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}")
        # print(f"epoch [{i + 1}/{num_epochs}] loss={loss.item()}")
        if loss.item() == 0.0:
            print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}\nstopped")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print(f"epoch [{i+1}/{num_epochs}] final loss {losses[-1]}")
    return losses


def I(size, device='cpu'):
    matrix = torch.zeros((size, size), dtype=torch.complex128, device=device).cuda()
    for i in range(size):
        matrix[i, i] = 1
    return matrix


def init(num_layers, num_qbits, num_epochs, device, qnn):
    x_qbits = num_qbits
    r_qbits = num_qbits

    print('prep')
    X = torch.tensor(np.array(uniform_random_data(2**x_qbits, 1, x_qbits, r_qbits)), device=device).cuda()
    U = random_unitary_matrix(x_qbits)
    # U = qnn.get_matrix_V()
    U_adj = U.conj().T
    r_I = I(2**r_qbits)
    U_adj_I = torch_tensor_product(torch.tensor(U_adj), r_I, device=device)
    U_adj_I = U_adj_I.to(device).cuda()

    optimizer = torch.optim.Adam(qnn.params, lr=0.1)

    print('train')
    starting_time = time.time()
    losses = train(X, qnn, U_adj_I, num_epochs, optimizer, r_I, device)
    total_time = time.time() - starting_time
    plot_loss(losses, num_qbits, num_layers)

    print(f"risk = {quantum_risk(U, qnn.get_matrix_V())}")
    if len(losses) == 0:
        return total_time, -1
    else:
        return total_time, np.array(losses).min()


def create_time_loss_plots():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_layers = [1] + list(range(5, 100, 5))
    qbits = [4]
    num_epochs = 100

    min_losses = []
    times = []
    for qbit in qbits:
        x_wires = list(range(qbit))
        for num_layer in num_layers:
            print(f"{qbit} qbits, {num_layer} layers, {num_epochs} epochs")
            qnn = TensorQNN(wires=x_wires, num_layers=num_layer, use_torch=True, device=device)
            training_time, min_loss = init(num_layer, qbit, num_epochs, device, qnn)
            print(f"min loss was {min_loss}")
            print(f"Training with {qbit} qubits and {num_layer} layers took {training_time}s")
            min_losses.append(min_loss)
            times.append(training_time)
        plt.plot(num_layers, min_losses)
        plt.xlabel('number of layers')
        plt.ylabel('min loss')
        plt.title(f'Min loss after {num_epochs} epochs for a {qbit} qubit system')
        plt.savefig(f'{directory}/plots/min_loss_{num_epochs}_epochs_{qbit}_qbits.png')
        plt.cla()
        plt.plot(num_layers, times)
        plt.xlabel('number of layers')
        plt.ylabel('runtime [s]')
        plt.title(f'Run time of {num_epochs} epochs for a {qbit} qubit system')
        plt.savefig(f'{directory}/plots/runtime_{num_epochs}_epochs_{qbit}_qbits.png')
        plt.cla()


def single_run():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    num_qbits= 6
    num_layers = 10
    num_epochs = 100
    x_wires = list(range(num_qbits))
    qnn = TensorQNN(wires=x_wires, num_layers=num_layers, use_torch=True, device=device)
    training_time, min_loss = init(num_layers=num_layers, num_qbits=num_qbits, num_epochs=num_epochs, device=device, qnn=qnn)
    print(f"{training_time}s")


def main():
    single_run()
    # create_time_loss_plots()


if __name__ == '__main__':
    main()
