from scipy.optimize import fsolve
from barren_plateau_hessian import get_param_indices
from copy import deepcopy
import numpy as np
from classic_training import cost_func as loss
import torch
from qnns.qnn import get_qnn
from data import random_unitary_matrix, uniform_random_data
from barren_plateau_hessian import *
import numpy as np
from itertools import product


def cartesian(arrays, out=None, dtype=np.float64):
    arrays = [np.asarray(x) for x in arrays]

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
        # out = np.array([[1e-1[[1e-16]*len(arrays) for _ in range(n)])
        # print(out)


    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def get_offset_list(list):
    list_of_offset = [el for el in product(list, repeat=4)]
    # print('Full offset', list_of_offset)

def compute_gradient(p_idx, qnn, X, y_conj):
    original_params = deepcopy(qnn.params)
    r = 1
    s = np.pi / (4 * r)
    qnn.params[p_idx[0]][p_idx[1:]] = original_params[p_idx[0]][p_idx[1:]] + s
    f_p_s = loss(X, y_conj, qnn)
    qnn.params[p_idx[0]][p_idx[1:]] = original_params[p_idx[0]][p_idx[1:]] - s
    f_m_s = loss(X, y_conj, qnn)
    sum = f_p_s - f_m_s
    return sum


def get_cost_function(qnn, X, U):
    param_indices = get_param_indices(qnn.params)
    y_conj = torch.matmul(U, X).conj()

    def cost_func(params):
        #try to get zero grad to find critical points
        for i in range(len(param_indices)):
            qnn.params[param_indices[i][0]][param_indices[i][1:]] = params[i]
        return [compute_gradient(p_idx, qnn, X, y_conj) for p_idx in param_indices]

    return cost_func


def neighbourhood_gradients(qnn, X, U, num_samples, sample_step_size):
    y_conj = torch.matmul(U, X).conj()
    param_indices = get_param_indices(qnn.params)
    sample_values = np.linspace(-num_samples*sample_step_size, num_samples*sample_step_size, num=2*num_samples+1, endpoint=True)
    neighbourhood_params = [el for el in product(sample_values, repeat=len(param_indices))]
    result = np.empty((len(neighbourhood_params), len(param_indices)))
    org_params = deepcopy(qnn.params)
    for n_p_idx in range(len(neighbourhood_params)):
        offsets = neighbourhood_params[n_p_idx]
        for i in range(len(param_indices)):
            qnn.params[param_indices[i]] = org_params[param_indices[i]] + offsets[i]
        for i in range(len(param_indices)):
            result[n_p_idx, i] = compute_gradient(param_indices[i], qnn, X, y_conj)
    qnn.params = org_params
    return result.reshape([len(sample_values)]*len(param_indices)+[len(param_indices)])


def neighbourhood_loss(qnn, X, U, num_samples, sample_step_size):
    y_conj = torch.matmul(U, X).conj()
    param_indices = get_param_indices(qnn.params)
    sample_values = np.linspace(-num_samples*sample_step_size, num_samples*sample_step_size, num=2*num_samples+1, endpoint=True, dtype=np.float64)
    neighbourhood_params = [el for el in product(sample_values, repeat=len(param_indices))]
    # print("neighbourhood_params:", neighbourhood_params)
    result = np.empty((len(neighbourhood_params),))
    org_params = deepcopy(qnn.params)
    for n_p_idx in range(len(neighbourhood_params)):
        offsets = neighbourhood_params[n_p_idx]
        for i in range(len(param_indices)):
            qnn.params[param_indices[i]] = org_params[param_indices[i]] + offsets[i]
        result[n_p_idx] = loss(X, y_conj, qnn)
    qnn.params = org_params
    return result.reshape([len(sample_values)]*len(param_indices))


def get_eigenvalues(x):
    #x = np.matrix(x)
    return np.linalg.eigvals(x)


def indefinite(x):
    eigenvalues = get_eigenvalues(x)
    if np.all(eigenvalues >=0):
        if np.all(eigenvalues > 0):
            return 'pos_def'
        else:
            return 'pos_semidef'
    elif np.all(eigenvalues <= 0 ):
        if np.all(eigenvalues < 0):
            return 'neg_def'
        else:
            return 'neg_semidef'
    else:
        return 'indef'


def main():
    x_qbits = 1
    num_layers = 1
    schmidt_rank = 1
    num_points = 2
    num_samples = 1
    sample_step_size = 1e-15
    grad_tol = 1e-12
    loss_tol = 1e-12
    QNNs = ['CudaPennylane']

    while True:
        U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128)
        X = torch.tensor(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, int(np.ceil(np.log2(schmidt_rank))))), dtype=torch.complex128)
        X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)

        for qnn_name in QNNs:
            qnn = get_qnn(qnn_name, list(range(x_qbits)), num_layers, device='cpu')
            qnn.params = qnn.params.detach()
            param_indices = get_param_indices(qnn.params)

            cost_func = get_cost_function(qnn, X, U)
            init_p = np.array([qnn.params[p_idx].item() for p_idx in get_param_indices(qnn.params)])
            out = fsolve(cost_func, init_p, xtol=1e-13, maxfev=1000)
            for i in range(len(out)):
                qnn.params[param_indices[i]] = out[i]

            print('test hessian for indefiniteness')
            hessian = calc_hessian(qnn, X, U)
            if(indefinite(hessian) == 'indef'):
                y_conj = torch.matmul(U, X).conj()
                p_loss = loss(X, y_conj, qnn).item()
                n_gradients = neighbourhood_gradients(qnn, X, U, num_samples, sample_step_size)
                n_loss = neighbourhood_loss(qnn, X, U, num_samples, sample_step_size)
                grad_mags = np.array([np.linalg.norm(grad) for grad in n_gradients])
                if (grad_mags <= grad_tol).all():
                    if (np.abs(n_loss - p_loss) <= loss_tol).all():
                        print("found plateau")
                        with open('./experimental_results/plateau/plateau_results.txt', 'a') as f:
                            f.write(f"plateau_type=strong, qnn_params={str(qnn.params.tolist()).replace(' ', '')}, unitary={str(U.tolist()).replace(' ', '')}, data_points={str(X.tolist()).replace(' ', '')}\n")
                            f.close()
                    print("found weak plateau")
                    with open('./experimental_results/plateau/plateau_results.txt', 'a') as f:
                        f.write(
                            f"plateau_type=weak, qnn_params={str(qnn.params.tolist()).replace(' ', '')}, unitary={str(U.tolist()).replace(' ', '')}, data_points={str(X.tolist()).replace(' ', '')}\n")
                        f.close()
                print("no plateau found at saddle point")


if __name__ == '__main__':
    torch.manual_seed(56789)
    np.random.seed(56789)
    main()
