from data import uniform_random_data, random_unitary_matrix
import time
import numpy as np
from utils import quantum_risk
import torch
import importlib

torch.manual_seed(4241)
np.random.seed(4241)


def quick_matmulvec(M, vec):
    size = M.shape[0] #quadratic, no care
    result = torch.zeros(vec.shape, dtype=torch.complex128)
    for i in range(0, vec.shape[0], size):
        result[i:i+size] = torch.matmul(M, vec[i:i+size])
    return result


def quick_matmulmat(A, B):
    """
    To do: A*(IxB) = X*(IxU.T)
    """
    size = B.shape[0]   # 2**x_qbit in case B=U.T
    result = torch.zeros(A.shape, dtype=torch.complex128)   # Size of X in case A=X
    for i in range(0, A.shape[0], size):
        for j in range(0, A.shape[1], size):
            result[i:i+size, j:j+size] = torch.matmul(A[i:i+size, j:j+size], B)
    return result


def cost_func(X, y_conj, qnn):
    """
    Compute cost function based on the circuit in Fig. 5 in Sharma et al.
    """
    cost = torch.zeros((1,))
    V = qnn.get_tensor_V()
    for idx in range(len(X)):
        state = quick_matmulvec(V, X[idx])
        state = torch.dot(y_conj[idx], state)
        cost += torch.square(state.real) + torch.square(state.imag)
        # cost += torch.square(torch.abs(torch.dot(el, state)))
    cost /= len(X)
    return 1 - cost


def train(X, unitary, qnn, num_epochs, optimizer, scheduler=None):
    losses = []
    y_conj = quick_matmulmat(X, torch.from_numpy(unitary.T)).conj()
    for i in range(num_epochs):
        loss = cost_func(X, y_conj, qnn)
        losses.append(loss.item())
        if i % 100 == 0:
            print(f"\tepoch [{i+1}/{num_epochs}] loss={loss.item()}")
        if loss.item() == 0.0:
            # print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}\nstopped")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0 and scheduler is not None:
            scheduler.step(loss.item())
            print(f"\tepoch [{i+1}/{num_epochs}] lr={scheduler.get_lr()}")
    print(f"\tepoch [{num_epochs}/{num_epochs}] final loss {losses[-1]}")
    return losses


def init(num_layers, num_qbits, schmidt_rank, num_points, num_epochs, lr, qnn_name, opt_name='Adam'):
    """
    Tensor training for QNN
    """
    starting_time = time.time()
    x_qbits = num_qbits
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))  # dont use all qubits for reference system
    x_wires = list(range(num_qbits))  # does not matter which qubits we are using, since we only want the matrix

    #construct QNNobject from qnn_name string
    qnn = getattr(importlib.import_module('qnn'), qnn_name)(wires=x_wires, num_layers=num_layers, use_torch=True)


    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))

    U = random_unitary_matrix(x_qbits)

    y_conj = quick_matmulmat(X, torch.from_numpy(U.T)).conj()

    if opt_name.lower() == 'sgd':
        optimizer = torch.optim.SGD
    else:
        optimizer = torch.optim.Adam

    if isinstance(qnn.params, list):
        optimizer = optimizer(qnn.params, lr=lr)
    else:
        optimizer = optimizer([qnn.params], lr=lr)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1)
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, min_lr=1e-10, verbose=True)
    prep_time = time.time() - starting_time
    print(f"\tPreparation with {num_qbits} qubits and {num_layers} layers took {prep_time}s")

    starting_time = time.time()
    losses = train(X, y_conj, qnn, num_epochs, optimizer, scheduler)
    train_time = time.time() - starting_time

    print(f"\trisk = {quantum_risk(U, qnn.get_matrix_V())}")
    return train_time, prep_time, losses
