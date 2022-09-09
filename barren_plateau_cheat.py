import torch
import numpy as np
from data import create_unitary_from_circuit, uniform_random_data
from qnns.qnn import get_qnn
from generate_experiments import get_optimizer, get_scheduler
import time
from metrics import quantum_risk
from classic_training import train, cost_func
from barren_plateau_solve import *


def process_execution():
    x_qbits = 6
    cheat = 2
    qnn_name = 'CudaPennylane'
    lr = 0.01
    device = 'cpu'
    num_layers = 2
    opt_name = 'adam'
    use_scheduler = True
    num_epochs = 2000
    scheduler_factor = 0.8
    scheduler_patience = 10

    schmidt_rank = 1
    num_points = 1

    # Do experiment
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    x_wires = list(range(x_qbits))

    U, unitary_qnn_params = create_unitary_from_circuit(qnn_name, x_wires, cheat, device='cpu')

    qnn = get_qnn(qnn_name, x_wires, num_layers, device=device)
    # torch.save(qnn.params, 'files_for_alex/qnn_params.pt')
    optimizer = get_optimizer(opt_name, qnn, lr)
    scheduler = get_scheduler(use_scheduler, optimizer, factor=scheduler_factor, patience=scheduler_patience)

    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
    X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
    starting_time = time.time()
    losses = train(X, U, qnn, num_epochs, optimizer, scheduler)
    train_time = time.time() - starting_time
    print(f"\tTraining took {train_time}s")
    # if r_idx == 2 and num_points_idx == 3:
    #     torch.save(qnn.params, './data/qnn_params.pt')
    #     torch.save(U, './data/U.pt')
    #     torch.save(X, './data/X.pt')

    risk = quantum_risk(U, qnn.get_matrix_V())
    # plot_line(X, qnn.params.detach(), unitary_qnn_params.detach())
    sample_neighbourhood(qnn, X, U)


def plot_line(X, qnn_params, u_params):
    import matplotlib.pyplot as plt
    x_qbits = list(range(6))
    qnn = get_qnn('CudaPennylane', x_qbits, 2, device='cpu')
    def line(alpha):
        return qnn_params + alpha * (u_params - qnn_params)
    loss = []
    u_qnn = get_qnn('CudaPennylane', x_qbits, 2, device='cpu')
    samples = np.linspace(-0.5, 1.5, num=2000, endpoint=True)
    for sample in samples:
        u_qnn.params = line(sample)
        y_conj = torch.matmul(u_qnn.get_tensor_V(), X)
        loss.append(cost_func(X, y_conj, qnn).item())
    loss = np.array(loss)
    samples = np.array(samples)
    plt.plot(samples, loss)
    plt.savefig('./plots/loss_map/cheat_u_loss.png')


def sample_neighbourhood_new(cheat_qnn, X, U):
        num_samples = 1
        static_samples = 1000
        sample_step_size = 1e-08
        grad_tol = 1e-08
        loss_tol = 1e-08
        qnn = cheat_qnn
        process_id = 0
        #qnn.params = qnn.params.detach()
        param_indices = get_param_indices(qnn.params)
        cost_func = get_cost_function(qnn, X, U)
        init_p = np.array([qnn.params[p_idx].item() for p_idx in get_param_indices(qnn.params)])
        out = fsolve(cost_func, init_p, xtol=1e-13, maxfev=1000)
        for i in range(len(out)):
            qnn.params[param_indices[i]] = out[i]

        print('{process_id} test hessian for indefiniteness')
        # hessian = calc_hessian(qnn, X, U)
        hessian = torch_calc_hessian(qnn, X, U)
        run_sampling(hessian, num_samples, static_samples, sample_step_size, grad_tol, loss_tol)

def sample_around_point(cheat_qnn, X, U):
    #test hessian with parameters of cheat qnn on last training step
    # with loss > 0

    hessian = torch_calc_hessian(cheat_qnn, X, U)
    run_sampling(hessian, num_samples, static_samples, sample_step_size, grad_tol, loss_tol)


def run_sampling(hessian, num_samples, static_samples, sample_step_size, grad_tol, loss_tol):
    if (indefinite(hessian) == 'indef'):
            y_conj = torch.matmul(U, X).conj()
            p_loss = loss(X, y_conj, qnn).item()

            param_indices = get_param_indices(qnn.params)
            sample_values = np.linspace(-num_samples * sample_step_size, num_samples * sample_step_size,
                                        num=2 * num_samples + 1, endpoint=True, dtype=np.float64)
            # grid sampling
            # neighbourhood_params = [el for el in product(sample_values, repeat=len(param_indices))]

            # static sampling

            neighbourhood_params = [
                np.random.uniform(-num_samples * sample_step_size, num_samples * sample_step_size,
                                  size=len(param_indices)) for i in range(static_samples)]
            # print(neighbourhood_params)
            n_gradients = torch_neighbourhood_gradients(qnn, X, U, num_samples, sample_step_size, param_indices,
                                                        sample_values, neighbourhood_params)
            n_loss = neighbourhood_loss(qnn, X, U, num_samples, sample_step_size, param_indices, sample_values,
                                        neighbourhood_params)
            grad_mags = np.array([np.linalg.norm(grad) for grad in n_gradients])
            if (grad_mags <= grad_tol).all():
                if (np.abs(n_loss - p_loss) <= loss_tol).all():
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


def main():
    process_execution()


if __name__ == '__main__':
    main()
