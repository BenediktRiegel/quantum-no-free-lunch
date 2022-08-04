import numpy as np

from config import *
from metrics import calc_avg_std_risk
import json
from logger import Writer
from qnns.qnn import get_qnn
import time
from data import *
import importlib
from classic_training import train
from metrics import quantum_risk
import matplotlib.pyplot as plt
import torch
from visualisation import *
from copy import deepcopy



def exp_basis_sharma(config, save_dir):
    """
    This procudes the results for plotting Figure 2 in the Sharma et al. Paper
    """
    # store config
    with open(save_dir + 'config.json', 'w') as f:
        json.dump(config, f)
        f.close()

    # logger = Logger(config['x_qbits'], config['num_points'], config['num_unitaries'],
    #                 config['num_training_data'])
    writer = Writer(save_dir + 'result.txt')
    all_risks = []
    for i in range(config['x_qbits'] + 1):
        rank = 2**i
        risk_list = []
        # print(f"rank {i+1}/{config['x_qbits'] + 1} (rank={rank})")
        r_qbits = i
        # logger.update_schmidt_rank(i)
        for num_points in range(1, config['num_points'] + 1):
            # logger.update_num_points(num_points)
            # print(f"num_points {num_points}/{config['num_points']}")
            risk, std = calc_avg_std_risk(rank, num_points, config['x_qbits'],
                                 r_qbits, config['num_unitaries'],
                                 config['num_layers'], config['num_training_data'],
                                 config['learning_rate'], config['num_epochs'],
                                 config['batch_size'],
                                 False, 0
                                 )
            # Store risks directly
            writer.append(f"{risk},{std}")
            risk_list.append([risk, std])
        all_risks.append(risk_list)
    all_risks_array = np.array(all_risks)

    # store risks
    np.save(save_dir + 'result.npy', all_risks_array)


def test_fig2():
    """
    This method generates results for Figure 2 in Sharma et al.
    """
    print("start experiment 1")
    #Fig.2 Paper Sharma et al.
    config = get_exp_one_qubit_unitary_config()
    print("config generated")
    exp_basis_sharma(config, './experimental_results/exp1/')


def test_fig3():
    """
    This method generates results for Figure 3 in Sharma et al.
    """
    print("start experiment 2")
    #Fig.3 Paper Sharma et al.
    #init config
    config = get_exp_six_qubit_unitary_config()
    print("config generated")

    # small test version
    exp_basis_sharma(config, './experimental_results/exp2/')


def test_simple_mean_std(upper_std):
    #store results for simply getting results from mean and std
    # it has to hold true: mean + upper_std <= max_rank


    print("start mean and std")
    config = gen_config(1, 1, 6, 6, 10, 10, 100, 1, 0, 0.01, 8, 120, True, 'SGD')
    print("config generated")

    save_dir = './experimental_results/exp_mean_std'
    # store config
    with open(save_dir + 'config.json', 'w') as f:
        json.dump(config, f)
        f.close()

    # logger = Logger(config['x_qbits'], config['num_points'], config['num_unitaries'],
    #                 config['num_training_data'])
    writer = Writer(save_dir + 'result.txt')
    all_risks = []

    for std in range(0, upper_std):
        #logger.update_num_points(num_points)
        # print(f"num_points {num_points}/{config['num_points']}")
        risk, _ = calc_avg_std_risk(config['rank'], config['num_points'], config['x_qbits'],
                                config['r_qbits'], config['num_unitaries'],
                                config['num_layers'], config['num_training_data'],
                                config['learning_rate'], config['num_epochs'],
                                config['batch_size'],
                                True, std)


        # Store risks directly
        writer.append(risk)
        all_risks.append(risk)
    all_risks_array = np.array(all_risks)

    # store risks
    np.save(save_dir + 'result.npy', all_risks_array)


def test_mean_std():
    #create more exhaustive experiment for mean and std
    #fix number go through all training pairs for the dataset
    #compare among different ranks by plotting different

    print("start mean and std")
    config = gen_config(1, 1, 6, 6, 10, 10, 100, 1, 0, 0.01, 8, 120, True, 'SGD')
    print("config generated")

    save_dir = './experimental_results/exp_mean_std'
    # store config
    with open(save_dir + 'config.json', 'w') as f:
        json.dump(config, f)
        f.close()

    # logger = Logger(config['x_qbits'], config['num_points'], config['num_unitaries'],
    #                 config['num_training_data'])
    writer = Writer(save_dir + 'result.txt')
    all_risks = []

    max_rank = 2**config['x_qbits']



    for std in range(0, max_rank - config['rank']):
        risk_list = []
        # print(f"rank {i+1}/{config['x_qbits'] + 1} (rank={rank})")
        #logger.update_schmidt_rank(i)
        for num_points in range(1, config['num_points'] + 1):
            #logger.update_num_points(num_points)
            # print(f"num_points {num_points}/{config['num_points']}")
            risk, _ = calc_avg_std_risk(config['rank'], config['num_points'], config['x_qbits'],
                                 config['r_qbits'], config['num_unitaries'],
                                 config['num_layers'], config['num_training_data'],
                                 config['learning_rate'], config['num_epochs'],
                                 config['batch_size'],
                                 True, std
                                 )
            # Store risks directly
            writer.append(risk)
            risk_list.append(risk)
        all_risks.append(risk_list)
    all_risks_array = np.array(all_risks)

    # store risks
    np.save(save_dir + 'result.npy', all_risks_array)


def generate_exp_data( x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name,
                       device, cheat, use_scheduler, opt_name, std=False, writer=None):
    """
    Generate experiment data
    
    Parameters
    ------------
    x_qbits: int
      Number of qubits for our system
    num_layers: int
      Number of layers for qnn
    num_epochs: int
      number of epochs to train qnn
    lr: float
        learning rate for training
    num_unitaries: int
        Number of unitaries to use for each step
    num_datasets: int
        Number of datasets to use for each step
    qnn_name: str
        Name of QNN to use
    device: str
        gpu vs cpu
    cheat: int
        indicates whether to use random unitary or 'cheat' unitary generated from circuit architecture.
        It also specifies the number of layers for the unitary.
    use_scheduler: boolean
        use scheduler for training or not
    """
    if writer:
        writer.append_line(f"x_qbits={x_qbits}, num_layers={num_layers}, num_epochs={num_epochs}, lr={lr}, "
                      f"num_unitaries={num_unitaries}, num_datasets={num_datasets}, qnn_name={qnn_name}, "
                      f"device={device}, cheat={cheat}, use_scheduler={use_scheduler}")
    complete_starting_time = time.time()
    r_list = list(range(x_qbits+1))
    num_datapoints = list(range(1, 2**x_qbits + 1))


    results = dict()
    for r_idx in range(len(r_list)):
        schmidt_rank = 2**r_list[r_idx]
        results[r_list[r_idx]] = []
        for num_points_idx in range(len(num_datapoints)):
            num_points = num_datapoints[num_points_idx]
            risks = []
            for unitary_idx in range(num_unitaries):
                r_qbits = int(np.ceil(np.log2(schmidt_rank)))
                x_wires = list(range(x_qbits))
                if cheat:
                    U, unitary_qnn_params = create_unitary_from_circuit(qnn_name, x_wires, cheat, device='cpu')
                    # torch.save(unitary_qnn_params, 'files_for_alex/unitary_qnn_params.pt')
                    # torch.save(U, 'files_for_alex/unitary_U.pt')
                else:
                    U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)
                for dataset_idx in range(num_datasets):
                    print(
                        f"Run: r [{r_idx + 1}/{len(r_list)}], no. points [{num_points_idx + 1}/{len(num_datapoints)}], "
                        f"U [{unitary_idx + 1}/{num_unitaries}], dataset [{dataset_idx + 1}/{num_datasets}]")
                    info_string = f"schmidt_rank={schmidt_rank}, num_points={num_points}, U=[{unitary_idx + 1}/{num_unitaries}], dataset=[{dataset_idx + 1}/{num_datasets}]"

                    qnn = get_qnn(qnn_name, x_wires, num_layers, device=device)
                    # torch.save(qnn.params, 'files_for_alex/qnn_params.pt')
                    if opt_name.lower() == 'adam':
                        optimizer = torch.optim.Adam
                    else:
                        optimizer = torch.optim.SGD

                    if isinstance(qnn.params, list):
                        optimizer = optimizer(qnn.params, lr=lr)
                    else:
                        optimizer = optimizer([qnn.params], lr=lr)
                    if use_scheduler:
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10,
                                                                               min_lr=1e-10,
                                                                               verbose=False)
                    else:
                        scheduler = None

                    if not std:
                        X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
                        X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
                        starting_time = time.time()
                        losses = train(X, U, qnn, num_epochs, optimizer, scheduler)
                        train_time = time.time() - starting_time
                        print(f"\tTraining took {train_time}s")

                        risk = quantum_risk(U, qnn.get_matrix_V())
                        risks.append(risk)
                    else:
                        losses = []
                        risk = []
                        train_time = []
                        max_rank = 2 ** x_qbits
                        for std in range(0, max_rank):
                            if min(schmidt_rank - 1, max_rank - schmidt_rank) <= 3*std:
                                continue
                            X = uniform_random_data_mean(schmidt_rank, std, num_points, x_qbits, r_qbits, max_rank)
                            X = torch.tensor(np.array(X), dtype=torch.complex128)
                            X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
                            starting_time = time.time()
                            loss_std = train(X, U, qnn, num_epochs, optimizer, scheduler)
                            train_time_std = time.time() - starting_time
                            print(f"\tTraining took {train_time_std}s")
                            risk_std = quantum_risk(U, qnn.get_matrix_V())
                            losses.append(loss_std)
                            risk.append(risk_std)
                            train_time.append(train_time_std)
                        risks.append(risk)
                    # Log everything
                    if writer:
                        writer.append_line(
                            info_string + f", losses={losses}, risk={risks[-1]}, train_time={train_time}")

            risks = np.array(risks)
            results[r_list[r_idx]].append(risks)

    # iterate over untrained unitaries
    zero_risks = []
    for unitary_idx in range(num_unitaries):
        x_wires = list(range(x_qbits))
        if cheat:
            U, _ = create_unitary_from_circuit(qnn_name, x_wires, cheat, device='cpu')
        else:
            U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)
        for dataset_idx in range(num_datasets):
            qnn = get_qnn(qnn_name, x_wires, num_layers, device='cpu')
            zero_risks.append(quantum_risk(U, qnn.get_matrix_V()))
    zero_risks = np.array(zero_risks)
    if writer:
        writer.append_line(f"zero_risks={zero_risks}")

    for r in r_list:
        results[r].insert(0, zero_risks)
    complete_time = time.time()-complete_starting_time
    print(f"Complete experiment took {complete_time}s")
    if writer:
        writer.append_line(f"complete_time={complete_time}")

    return results


def exp(x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name, device, cheat, use_scheduler, optimizer, std=False, writer=None):
    """
    Start experiments, including generation of data as well as plot

    """
    results = generate_exp_data(x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name, device, cheat, use_scheduler, optimizer, std=std, writer=writer)
    num_datapoints = list(range(0, 2**x_qbits + 1))
    r_list = list(range(x_qbits + 1))
    generate_risk_plot(results, num_datapoints, x_qbits, r_list)


def gradient_3D_plot(U=None):
    from classic_training import cost_func
    # Parameters
    x_qbits = 2
    r = 0
    num_points = 4
    qnn_name = 'CudaPlataeu'
    device = 'cpu'
    num_layers = 1

    param_idx1 = (0, 0)
    param_idx2 = (1, 0)
    param_idx1_range = (0, 2*np.pi, 0.1)   # Lower limit, upper limit, step size
    param_idx2_range = (0, 2*np.pi, 0.1)   # Lower limit, upper limit, step size

    # qnn
    x_wires = list(range(x_qbits))
    qnn = get_qnn(qnn_name, x_wires, num_layers, device='cpu')
    qnn.params = qnn.params.detach()

    # Unitray U
    if U is None:
        # U, U_params = create_unitary_from_circuit(qnn_name, x_wires, num_layers, device=device)
        # qnn.params = U_params
        U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)


    # Dataset X
    schmidt_rank = 2**r
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
    X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)

    # Labels y conjugated
    y_conj = torch.matmul(U, X).conj()

    param1_list = np.linspace(param_idx1_range[0], param_idx1_range[1], num=int(param_idx1_range[1]/param_idx1_range[2]), endpoint=False)
    param2_list = np.linspace(param_idx2_range[0], param_idx2_range[1], num=int(param_idx2_range[1]/param_idx2_range[2]), endpoint=False)
    result = []
    params = deepcopy(qnn.params).detach()
    for param2 in param2_list:
        result.append([])
        params[param_idx2] = param2
        for param1 in param1_list:
            # print(qnn.params)
            # opt = torch.optim.Adam([qnn.params], lr= 0.1)
            # opt.zero_grad()
            params[param_idx1] = param1
            qnn.params = params.clone().detach().requires_grad_(True)
            loss = cost_func(X, y_conj, qnn, device = device)
            #print('backprop')
            loss.backward()

            # print('Gradients', np.array(qnn.params.grad))
            result[-1].append(np.linalg.norm(np.array(qnn.params.grad)))    #cost_func(X,y_conj,qnn, device)

    import plotly.graph_objects as go
    import plotly

    result = np.array(result)

    fig = go.Figure(data=[go.Surface(x=param1_list, y=param2_list, z=result)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Gradient Map', autosize=True)

    plotly.offline.plot(fig, filename='./plots/loss_map/gradient_map.html')
    # fig.show()




def map_loss_function(U=None):
    from classic_training import cost_func
    # Parameters
    x_qbits = 2
    r = 0
    num_points = 4
    qnn_name = 'CudaPlataeu'
    device = 'cpu'
    num_layers = 1

    param_idx1 = (0, 0)
    param_idx2 = (1, 0)
    param_idx1_range = (0, 2*np.pi, 0.1)   # Lower limit, upper limit, step size
    param_idx2_range = (0, 2*np.pi, 0.1)   # Lower limit, upper limit, step size

    # qnn
    x_wires = list(range(x_qbits))
    qnn = get_qnn(qnn_name, x_wires, num_layers, device='cpu')
    qnn.params = qnn.params.detach()


    # Unitray U
    if U is None:
        # U, U_params = create_unitary_from_circuit(qnn_name, x_wires, num_layers, device=device)
        # qnn.params = U_params
        U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)


    # Dataset X
    schmidt_rank = 2**r
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
    X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)

    # Labels y conjugated
    y_conj = torch.matmul(U, X).conj()

    param1_list = np.linspace(param_idx1_range[0], param_idx1_range[1], num=int(param_idx1_range[1]/param_idx1_range[2]), endpoint=False)
    param2_list = np.linspace(param_idx2_range[0], param_idx2_range[1], num=int(param_idx2_range[1]/param_idx2_range[2]), endpoint=False)
    result = []
    for param2 in param2_list:
        result.append([])
        qnn.params[param_idx2] = param2
        for param1 in param1_list:
            qnn.params[param_idx1] = param1
            result[-1].append(cost_func(X, y_conj, qnn, device=device))

    import plotly.graph_objects as go
    import plotly

    result = np.array(result)

    fig = go.Figure(data=[go.Surface(x=param1_list, y=param2_list, z=result)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Loss Map', autosize=True)

    plotly.offline.plot(fig, filename='./plots/loss_map/loss_map.html')
    # fig.show()



if __name__ == '__main__':
    # U = torch.tensor(random_unitary_matrix(2), dtype=torch.complex128, device='cpu')
    # map_loss_function(U=U)
    # gradient_3D_plot(U=U)

    exp(4, 40, 1000, 0.1, 1, 1, 'CudaPennylane', 'cpu', None, True, 'Adam', std=True,
        writer=Writer('./experimental_results/4_qubit_exp_std.txt'))
