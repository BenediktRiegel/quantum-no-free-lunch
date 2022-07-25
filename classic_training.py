import time

import numpy as np
import torch
from quantum_gates import quick_matmulmat, quick_matmulvec

torch.manual_seed(4241)
np.random.seed(4241)

cost = None


def init(device='cpu'):
    global cost
    cost = torch.zeros((1,), device=device)


def cost_func(X, y_conj, qnn, device='cpu'):
    """
    Compute cost function based on the circuit in Fig. 5 in Sharma et al.
    """
    global cost
    cost = 0
    V = qnn.get_tensor_V()
    for idx in range(len(X)):
        state = quick_matmulvec(V, X[idx])
        state = torch.dot(y_conj[idx], state)
        cost += torch.square(state.real) + torch.square(state.imag)
    cost /= len(X)
    return 1 - cost


def train(X, unitary, qnn, num_epochs, optimizer, scheduler=None, device='cpu'):
    losses = []
    y_conj = quick_matmulmat(X, unitary.T, device=device).conj()
    for i in range(num_epochs):
        loss = cost_func(X, y_conj, qnn, device=device)
        losses.append(loss.item())
        if i % 100 == 0:
            print(f"\tepoch [{i+1}/{num_epochs}] loss={loss.item()}")
        if loss.item() == 0.0:
            # print(f"epoch [{i+1}/{num_epochs}] loss={loss.item()}\nstopped")
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(loss.item())
            # print(f"\tepoch [{i + 1}/{num_epochs}] lr={scheduler.get_lr()}")
    print(f"\tepoch [{num_epochs}/{num_epochs}] final loss {losses[-1]}")
    return losses
