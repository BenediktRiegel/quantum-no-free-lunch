import time

import numpy as np
import torch

# torch.manual_seed(4241)
# np.random.seed(4241)

def identity(value):
    return value


def cost_func(X, y_conj, qnn, device='cpu'):
    """
    Compute cost function based on the circuit in Fig. 5 in Sharma et al.
    """
    V = qnn.get_tensor_V()
    dot_products = torch.sum(torch.mul(torch.matmul(V, X), y_conj), dim=[1, 2])
    cost = (torch.sum(torch.square(dot_products.real)) + torch.sum(torch.square(dot_products.imag))) / X.shape[0]
    return 1 - cost


def train(X, unitary, qnn, num_epochs, optimizer, scheduler=None, device='cpu', cost_modification=None):
    if cost_modification is None:
        cost_modification = identity
    losses = []
    y_conj = torch.matmul(unitary, X).conj()
    i = 0
    for i in range(num_epochs):
        loss = cost_modification(cost_func(X, y_conj, qnn, device=device))
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
    print(f"\tepoch [{i+1}/{num_epochs}] final loss {losses[-1]}")
    return losses
