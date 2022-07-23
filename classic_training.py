import numpy as np
import torch

torch.manual_seed(4241)
np.random.seed(4241)

cost = None


def init(device='cpu'):
    global cost
    cost = torch.zeros((1,), device=device)


def quick_all_matmul_vec(M, X, device='cpu'):
    # global matmulvec_result
    matmulvec_result = torch.empty(X.shape, dtype=torch.complex128, device=device)
    size = M.shape[0]
    for idx in range(len(X)):
        for i in range(0, matmulvec_result.shape[1], size):
            matmulvec_result[idx, i:i+size] = torch.matmul(M, X[idx, i:i+size])
    return matmulvec_result


def quick_matmulvec(M, vec, device='cpu'):
    matmulvec_result = torch.empty(vec.shape, dtype=torch.complex128, device=device)
    size = M.shape[0] #quadratic, no care

    for i in range(0, vec.shape[0], size):
        matmulvec_result[i:i+size] = torch.matmul(M, vec[i:i+size])
    return matmulvec_result


def quick_matmulmat(A, B, device='cpu'):
    """
    To do: A*(IxB) = X*(IxU.T)
    """
    matmulmat_result = torch.empty(A.shape, dtype=torch.complex128, device=device)
    size = B.shape[0]   # 2**x_qbit in case B=U.T
    for i in range(0, A.shape[0], size):
        for j in range(0, A.shape[1], size):
            matmulmat_result[i:i+size, j:j+size] = torch.matmul(A[i:i+size, j:j+size], B)
    return matmulmat_result


def cost_func(X, y_conj, qnn, device='cpu'):
    """
    Compute cost function based on the circuit in Fig. 5 in Sharma et al.
    """
    global cost
    cost = 0
    # cost = torch.zeros((1,), device=device)
    V = qnn.get_tensor_V()
    # for idx in range(len(X)):
    #     state = quick_matmulvec(V, X[idx], device=device)
    #     state = torch.dot(y_conj[idx], state)
    #     cost += torch.square(state.real) + torch.square(state.imag)
        # cost += torch.square(torch.abs(torch.dot(el, state)))
    all_states = quick_all_matmul_vec(V, X, device=device)
    for idx in range(len(X)):
        state = torch.dot(y_conj[idx], all_states[idx])
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
