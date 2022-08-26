import numpy as np
import torch
from copy import deepcopy
from qnns.qnn import get_qnn
from data import uniform_random_data, random_unitary_matrix
from classic_training import cost_func
from torch.autograd.functional import hessian


def get_param_indices(params):
    indices = []
    if isinstance(params, torch.Tensor):
        if len(params.shape) == 1:
            return [(i,) for i in range(params.shape[0])]
    for i in range(len(params)):
        indices += [(i,)+el for el in get_param_indices(params[i])]
    return indices


def get_shift_derv(x_i, x_j, qnn, X, y_conj):
    original_params = deepcopy(qnn.params)
    r = 1
    s = np.pi/(4*r)
    if x_i == x_j:
        f = cost_func(X, y_conj, qnn)
        qnn.params[x_i[0]][x_i[1:]] = original_params[x_i[0]][x_i[1:]] + 2*s
        f_p_2si = cost_func(X, y_conj, qnn)
        qnn.params[x_i[0]][x_i[1:]] = original_params[x_i[0]][x_i[1:]] - 2*s
        f_m_2si = cost_func(X, y_conj, qnn)
        sum = f_p_2si - 2*f + f_m_2si
    else:
        # xi + si and xj + sj
        qnn.params[x_i[0]][x_i[1:]] = original_params[x_i[0]][x_i[1:]] + s
        qnn.params[x_j[0]][x_j[1:]] = original_params[x_j[0]][x_j[1:]] + s
        f_p_si_p_sj = cost_func(X, y_conj, qnn)
        # xi + si and xj - sj
        qnn.params[x_j[0]][x_j[1:]] = original_params[x_j[0]][x_j[1:]] - s
        f_p_si_m_sj = cost_func(X, y_conj, qnn)
        # xi - si and xj - sj
        qnn.params[x_i[0]][x_i[1:]] = original_params[x_i[0]][x_i[1:]] - s
        f_m_si_m_sj = cost_func(X, y_conj, qnn)
        # xi - si and xj + sj
        qnn.params[x_j[0]][x_j[1:]] = original_params[x_j[0]][x_j[1:]] + s
        f_m_si_p_sj = cost_func(X, y_conj, qnn)
        # sum = second derivative
        sum = f_p_si_p_sj - f_p_si_m_sj - f_m_si_p_sj + f_m_si_m_sj

    qnn.params = original_params

    return sum.item()


def calc_hessian(qnn, X, U):
    if isinstance(qnn.params, list):
        qnn.params = [el.detach() for el in qnn.params]
    else:
        qnn.params = qnn.params.detach()
    indices = get_param_indices(qnn.params)
    hessian_matrix = np.zeros((len(indices), len(indices)))
    # qnn.params = List manchmal n Tensor und die haben mehrere Dimensionen
    #define shift
    y_conj = torch.matmul(U, X).conj()

    for x_i in range(len(indices)):
        for x_j in range(x_i):
            hessian_matrix[x_i][x_j] = hessian_matrix[x_j][x_i]
        for x_j in range(x_i, len(indices)):
            derv = get_shift_derv(indices[x_i], indices[x_j], qnn, X, y_conj)
            hessian_matrix[x_i][x_j] = derv
    return hessian_matrix.T


def torch_calc_hessian(qnn, X, U):
    params = qnn.params
    y_conj = torch.matmul(U, X).conj()
    def func(params):
        qnn.params = params
        return cost_func(X, y_conj, qnn)
    return hessian(func, params)


def torch_gradients(qnn, X, y_conj):
    loss = cost_func(X, y_conj, qnn)
    qnn.params.retain_grad()
    loss.backward()
    gradients = qnn.params.grad
    return gradients


if __name__ == '__main__':
    x_qbits = 1
    num_layers = 1
    schmidt_rank = 1
    num_points = 1
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))

    qnn = get_qnn('CudaPennylane', range(x_qbits), 1, 'cpu')
    qnn.params = qnn.params.detach()

    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
    X = X.reshape((X.shape[0], int(X.shape[1] / 2**x_qbits), 2**x_qbits)).permute(0, 2, 1)
    X[0, 0, 0] = 1
    X[0, 1, 0] = 0
    print(X)

    U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device='cpu')
    U = torch.eye(U.shape[0], dtype=torch.complex128)
    print(U)

    indices = get_param_indices(qnn.params)
    for idx in indices:
        qnn.params[idx[0]][idx[1:]] = 0

    print(qnn.params)
    print(qnn.get_tensor_V())

    hessian = calc_hessian(qnn, X, U)
    print(hessian)
