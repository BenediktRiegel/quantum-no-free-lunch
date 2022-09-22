from scipy.optimize import fsolve
from classic_training import cost_func as loss
from barren_plateau_hessian import *
from itertools import product
from concurrent.futures import ProcessPoolExecutor



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


def neighbourhood_gradients(qnn, X, U, num_samples, sample_step_size, param_indices, sample_values, neighbourhood_params):
    y_conj = torch.matmul(U, X).conj()

    result = np.empty((len(neighbourhood_params), len(param_indices)))
    org_params = deepcopy(qnn.params)
    for n_p_idx in range(len(neighbourhood_params)):
        offsets = neighbourhood_params[n_p_idx]
        for i in range(len(param_indices)):
            qnn.params[param_indices[i]] = org_params[param_indices[i]] + offsets[i]
        for i in range(len(param_indices)):
            result[n_p_idx, i] = compute_gradient(param_indices[i], qnn, X, y_conj)
    qnn.params = org_params
    #return result.reshape([len(sample_values)]*len(param_indices)+[len(param_indices)])
    return result


def torch_neighbourhood_gradients(qnn, X, U, num_samples, sample_step_size, param_indices, sample_values, neighbourhood_params):
    y_conj = torch.matmul(U, X).conj()

    #print(len(neighbourhood_params))
    result = np.empty((len(neighbourhood_params), len(param_indices)))
    org_params = qnn.params
    for n_p_idx in range(len(neighbourhood_params)):
        offsets = neighbourhood_params[n_p_idx]
        new_params = torch.empty(org_params.shape, dtype=torch.float64)
        for i in range(len(param_indices)):
            new_params[param_indices[i][0]][param_indices[i][1:]] = org_params[param_indices[i]] + offsets[i]
        # qnn.params = torch.tensor(new_params, requires_grad=True, dtype=torch.float64)
        qnn.params = new_params.requires_grad_(True)
        grads = torch_gradients(qnn, X, y_conj)
        for i in range(len(param_indices)):
            result[n_p_idx][i] = np.array(grads[param_indices[i][0]][param_indices[i][1:]])
    qnn.params = org_params
    # [len(sample_values), ...,len(sample_values), len(param_indices)]
    #return result.reshape([len(sample_values)]*len(param_indices)+[len(param_indices)]

    return result


def neighbourhood_loss(qnn, X, U, num_samples, sample_step_size, param_indices, sample_values, neighbourhood_params):
    y_conj = torch.matmul(U, X).conj()

    # print("neighbourhood_params:", neighbourhood_params)
    result = np.empty((len(neighbourhood_params),))
    org_params = qnn.params
    for n_p_idx in range(len(neighbourhood_params)):
        offsets = neighbourhood_params[n_p_idx]
        new_params = torch.empty(qnn.params.shape, dtype=torch.float64)
        for i in range(len(param_indices)):
            new_params[param_indices[i][0]][param_indices[i][1:]] = org_params[param_indices[i][0]][param_indices[i][1:]] + offsets[i]
        qnn.params = new_params
        result[n_p_idx] = loss(X, y_conj, qnn)
    qnn.params = org_params
    #return result.reshape([len(sample_values)]*len(param_indices))
    return result.flatten()

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
    
    
def simple_plateau_search(process_id, x_qbits, num_layers, sample_step_size, grad_tol, 
                          loss_tol, QNNs, schmidt_rank, num_points, num_samples, static_samples):
    for element in product(x_qbits, num_layers, sample_step_size, grad_tol, loss_tol, QNNs):
        x_qbits_prod = element[0]
        num_layers_prod = element[1]
        sample_step_size_prod = element[2]
        grad_tol_prod = element[3]
        loss_tol_prod = element[4]
        qnn_name = element[5]

        U = torch.tensor(random_unitary_matrix(x_qbits_prod), dtype=torch.complex128)
        X = torch.tensor(
            np.array(uniform_random_data(schmidt_rank, num_points, x_qbits_prod, int(np.ceil(np.log2(schmidt_rank))))),
            dtype=torch.complex128)
        X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)

        qnn = get_qnn(qnn_name, list(range(x_qbits_prod)), num_layers_prod, device='cpu')

        if isinstance(qnn.params, list):
            qnn.params = [el.detach() for el in qnn.params]
        else:
            qnn.params = qnn.params.detach()

        param_indices = get_param_indices(qnn.params)

        cost_func = get_cost_function(qnn, X, U)
        init_p = np.array([qnn.params[p_idx[0]][p_idx[1:]].item() for p_idx in get_param_indices(qnn.params)])
        out = fsolve(cost_func, init_p, xtol=1e-13, maxfev=1000)
        for i in range(len(out)):
            qnn.params[param_indices[i][0]][param_indices[i][1:]] = out[i]

        print(f'{process_id}: test hessian for indefiniteness')
        # hessian = calc_hessian(qnn, X, U)
        hessian = torch_calc_hessian(qnn, X, U)
        if (indefinite(hessian) == 'indef'):
            y_conj = torch.matmul(U, X).conj()
            p_loss = loss(X, y_conj, qnn).item()

            param_indices = get_param_indices(qnn.params)
            sample_values = np.linspace(-num_samples * sample_step_size_prod, num_samples * sample_step_size_prod,
                                        num=2 * num_samples + 1, endpoint=True, dtype=np.float64)
            # grid sampling
            # neighbourhood_params = [el for el in product(sample_values, repeat=len(param_indices))]

            # static sampling

            neighbourhood_params = [
                np.random.uniform(-num_samples * sample_step_size_prod, num_samples * sample_step_size_prod,
                                  size=len(param_indices)) for i in range(static_samples)]
            # print(neighbourhood_params)
            n_gradients = torch_neighbourhood_gradients(qnn, X, U, num_samples, sample_step_size_prod, param_indices,
                                                        sample_values, neighbourhood_params)
            n_loss = neighbourhood_loss(qnn, X, U, num_samples, sample_step_size_prod, param_indices, sample_values,
                                        neighbourhood_params)
            grad_mags = np.array([np.linalg.norm(grad) for grad in n_gradients])
            if (grad_mags <= grad_tol_prod).all():
                if (np.abs(n_loss - p_loss) <= loss_tol_prod).all():
                    print("found plateau")
                    with open(f'./experimental_results/plateau/plateau_results_{process_id}.txt', 'a') as f:
                        f.write(
                            f"plateau_type=strong, qnn_params={str(qnn.params.tolist()).replace(' ', '')}, unitary={str(U.tolist()).replace(' ', '')}, data_points={str(X.tolist()).replace(' ', '')}, QNN architecture={str(qnn_name)}, x_qbits={str(x_qbits_prod)}, num_layers={str(num_layers_prod)}, sample_step_size={str(sample_step_size_prod)}, grad_tol={str(grad_tol_prod)}, loss_tol={str(loss_tol_prod)}\n")
                        f.close()
                else:
                    with open(f'./experimental_results/plateau/plateau_results_{process_id}.txt', 'a') as f:
                        f.write(
                            f"plateau_type=weak, qnn_params={str(qnn.params.tolist()).replace(' ', '')}, unitary={str(U.tolist()).replace(' ', '')}, data_points={str(X.tolist()).replace(' ', '')}, QNN architecture={str(qnn_name)}, x_qbits={str(x_qbits_prod)}, num_layers={str(num_layers_prod)}, sample_step_size={str(sample_step_size_prod)}, grad_tol={str(grad_tol_prod)}, loss_tol={str(loss_tol_prod)}\n")
                        f.close()
                print(f"{process_id}: found weak plateau")
            print(f"{process_id}: no plateau found at saddle point")


def process_execution(process_id):
    # x_qbits = [1,2]
    x_qbits = [4]
    num_layers = [3,4,5]
    schmidt_rank = 1
    num_points = 2
    num_samples = 1
    static_samples = 1000
    sample_step_size = [1e-12, 1e-10, 1e-08]
    grad_tol = [1e-12, 1e-10, 1e-08]
    loss_tol = [1e-12, 1e-10, 1e-08]
    # QNNs = ['CudaPennylane']
    QNNs = ['CudaCircuit6']
    num_trials = 100

    for current_trials in range(num_trials):
        simple_plateau_search(process_id, x_qbits, num_layers, sample_step_size, grad_tol,
                              loss_tol, QNNs, schmidt_rank, num_points, num_samples, static_samples)
    # for current_trials in range(num_trials):
    #     cheat_plateau_search()


def main():
    num_processes = 8
    ppe = ProcessPoolExecutor(max_workers=num_processes)
    worker_args = list(range(num_processes))
    results = ppe.map(process_execution, worker_args)
    for res in results:
        print(res)


if __name__ == '__main__':
    # torch.manual_seed(56789)
    # np.random.seed(56789)
    main()
