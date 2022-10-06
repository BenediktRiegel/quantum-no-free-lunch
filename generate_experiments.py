import numpy as np
import torch

from config import *
from logger import Writer, log_line_to_dict, check_dict_for_attributes
from classic_training import train
from visualisation import *
from data import *
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import glob
import os


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


def get_optimizer(opt_name, qnn, lr):
    if opt_name.lower() == 'adam':
        optimizer = torch.optim.Adam
    else:
        optimizer = torch.optim.SGD

    if isinstance(qnn.params, list):
        optimizer = optimizer(qnn.params, lr=lr)
    else:
        optimizer = optimizer([qnn.params], lr=lr)

    return optimizer


def get_scheduler(use_scheduler, optimizer,factor=0.8, patience=3, verbose=False):
    if use_scheduler:
        # some old values: factor=0.8, patience=10, min_lr=1e-10, verbose=False
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=1e-10, verbose=verbose)
    else:
        scheduler = None

    return scheduler


def process_execution(args):
    (process_id, num_processes, writer, idx_file_path, exp_file_path,
     x_qbits, cheat, qnn_name, lr, device, num_layers, opt_name, use_scheduler, num_epochs,
     scheduler_factor, scheduler_patience, small_std, cost_modification) = args
    print(f'Awesome process with id {process_id}')
    if not exists(idx_file_path):
        raise ValueError('idx_file does not exist')
    current_idx = process_id - num_processes
    with open(idx_file_path, 'r') as idx_file:
        first_line = idx_file.readline().replace('\n', '')
        current_idx = int(first_line)
        idx_file.close()
    current_idx += num_processes

    line_dict = None
    attributes = dict(schmidt_rank='*', num_points='*', std='*')
    while True:
        with open(exp_file_path, 'r') as exp_file:
            current_line = exp_file.readline()
            for i in range(current_idx-1):
                current_line = exp_file.readline()
            line_dict = log_line_to_dict(current_line)
            exp_file.close()

        if line_dict is None or not check_dict_for_attributes(line_dict, attributes):
            return

        # line_dict entries: schmidt_rank, num_points, std, unitary_idx, dataset_idx
        schmidt_rank = line_dict['schmidt_rank']
        num_points = line_dict['num_points']
        std = line_dict['std']

        # Do experiment
        r_qbits = int(np.ceil(np.log2(schmidt_rank)))
        x_wires = list(range(x_qbits))

        if cheat:
            U, unitary_qnn_params = create_unitary_from_circuit(qnn_name, x_wires, cheat, device='cpu')
        else:
            U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)
        print(f"Run: current_exp_idx={current_idx}")
        info_string = f"schmidt_rank={schmidt_rank}, num_points={num_points}"

        qnn = get_qnn(qnn_name, x_wires, num_layers, device=device)
        # torch.save(qnn.params, 'files_for_alex/qnn_params.pt')
        optimizer = get_optimizer(opt_name, qnn, lr)
        scheduler = get_scheduler(use_scheduler, optimizer, factor=scheduler_factor, patience=scheduler_patience)

        if std == 0:
            X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
            X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)
            # if r_idx == 2 and num_points_idx == 3:
            #     torch.save(qnn.params, './data/qnn_params.pt')
            #     torch.save(U, './data/U.pt')
            #     torch.save(X, './data/X.pt')
            starting_time = time.time()
            losses = train(X, U, qnn, num_epochs, optimizer, scheduler, cost_modification=cost_modification)
            train_time = time.time() - starting_time
            print(f"\tTraining took {train_time}s")

            risk = quantum_risk(U, qnn.get_matrix_V())
            # Log everything
            if writer:
                losses_str = str(losses).replace(' ', '')
                qnn_params_str = str(qnn.params.tolist()).replace(' ', '')
                u_str = str(qnn.params.tolist()).replace(' ', '')
                writer.append_line(
                    info_string + f", std={0}, losses={losses_str}, risk={risk}, train_time={train_time}, qnn={qnn_params_str}, unitary={u_str}"
                )
        else:
            if not small_std:
                losses = []
                risk = []
                train_time = []
                max_rank = 2 ** x_qbits
                X, final_std, final_mean = uniform_random_data_mean(schmidt_rank, std, num_points, x_qbits, r_qbits, max_rank)
                if final_std != 0:
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
                    # Log everything
                    if writer:
                        losses_str = str(losses).replace(' ', '')
                        qnn_params_str = str(qnn.params.tolist()).replace(' ', '')
                        u_str = str(qnn.params.tolist()).replace(' ', '')
                        writer.append_line(
                            info_string + f", std={final_std}, mean={final_mean}, losses={losses_str}, risk={risk_std}, train_time={train_time}, qnn={qnn_params_str}, unitary={u_str}"
                        )
                else:
                    print(f"\nfinal_std={final_std} so we skip\n")
            else:

                X, std, mean = uniform_random_data_mean_pair(schmidt_rank, std, num_points, x_qbits)
                X = torch.tensor(np.array(X), dtype=torch.complex128)
                X = X.reshape((X.shape[0], int(X.shape[1] / U.shape[0]), U.shape[0])).permute(0, 2, 1)

                starting_time = time.time()
                losses = train(X, U, qnn, num_epochs, optimizer, scheduler)
                train_time = time.time() - starting_time
                print(f"\tTraining took {train_time}s")
                risk = quantum_risk(U, qnn.get_matrix_V())
                # Log everything
                if writer:
                    losses_str = str(losses).replace(' ', '')
                    qnn_params_str = str(qnn.params.tolist()).replace(' ', '')
                    u_str = str(qnn.params.tolist()).replace(' ', '')
                    writer.append_line(
                        info_string + f", std={std}, mean={mean}, losses={losses}, risk={risk}, train_time={train_time}, qnn={qnn_params_str}, unitary={u_str}"
                    )
        current_idx += num_processes
        with open(idx_file_path, 'w') as idx_file:
            idx_file.write(str(current_idx))
            idx_file.close()


def generate_exp_data(x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name,
                       device, cheat, use_scheduler, opt_name, scheduler_factor=0.8, scheduler_patience=3, std=False,
                       writer_path=None, num_processes=1, run_type='new', small_std=False,
                      schmidt_ranks=None, num_datapoints=None, cost_modification=None):
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
    #ProcessPoolExecutor
    torch.multiprocessing.set_start_method('spawn')
    idx_file_dir = writer_path
    if run_type != 'continue':
        filelist = glob.glob(os.path.join(idx_file_dir, "process_idx_*.txt"))
        for f in filelist:
            os.remove(f)


    writers = [None]*num_processes
    if writer_path:
        writers = [Writer(writer_path+f"result_{process_id}.txt", delete=(run_type != 'continue')) for process_id in range(num_processes)]
        for writer in writers:
            writer.append_line(f"x_qbits={x_qbits}, num_layers={num_layers}, num_epochs={num_epochs}, lr={lr}, "
                          f"num_unitaries={num_unitaries}, num_datasets={num_datasets}, qnn_name={qnn_name}, "
                          f"device={device}, cheat={cheat}, use_scheduler={use_scheduler}")

    exp_file_path = gen_exp_file(x_qbits, num_unitaries, num_datasets, std, small_std, schmidt_ranks, num_datapoints)
    complete_starting_time = time.time()

    # (process_id, num_processes, writer, idx_file_path, exp_file_path,
    #  x_qbits, cheat, qnn_name, device, num_layers, opt_name, use_scheduler, num_epochs)

    ppe = ProcessPoolExecutor(max_workers=num_processes)
    worker_args = []
    for process_id in range(num_processes):
        idx_file_path = idx_file_dir + f'process_idx_{process_id}.txt'
        if not exists(idx_file_path):
            with open(idx_file_path, 'w') as idx_file:
                idx_file.write(str(process_id-num_processes))
                idx_file.close()
        worker_args.append((process_id, num_processes, writers[process_id], idx_file_path, exp_file_path, x_qbits,
                            cheat, qnn_name, lr, device, num_layers, opt_name, use_scheduler, num_epochs,
                            scheduler_factor, scheduler_patience, small_std, cost_modification))
    results = ppe.map(process_execution, worker_args)
    for res in results:
        print(res)
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
    if writers:
        writers[0].append_line(f"zero_risks={zero_risks}")

    complete_time = time.time()-complete_starting_time
    print(f"Complete experiment took {complete_time}s")
    for writer in writers:
        writer.append_line(f"complete_time={complete_time}")


def exp(x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name, device, cheat, use_scheduler,
        optimizer, scheduler_factor=0.8, scheduler_patience=3, std=False, writer_path=None, num_processes=1, run_type='continue',
        small_std=False,
        schmidt_ranks=None, num_datapoints=None, cost_modification=None):
    """
    Start experiments, including generation of data as well as plot

    """

    generate_exp_data(
        x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name, device, cheat, use_scheduler,
        optimizer, scheduler_factor=scheduler_factor, scheduler_patience=scheduler_patience, std=std,
        writer_path=writer_path, num_processes=num_processes, run_type=run_type, small_std=small_std,
        schmidt_ranks=schmidt_ranks, num_datapoints=num_datapoints, cost_modification=cost_modification,
    )
    # num_datapoints = list(range(0, 2**x_qbits + 1))
    # r_list = list(range(x_qbits + 1))
    # generate_risk_plot(results, num_datapoints, x_qbits, r_list)


def gradient_3D_plot(U=None, X=None, func=None, name='gradient_map'):
    from classic_training import cost_func
    # Parameters
    x_qbits = 1
    r = 0
    num_points = 2
    qnn_name = 'CudaPennylane'
    device = 'cpu'
    num_layers = 1

    param_idx1 = (0, 0, 0)
    param_idx2 = (0, 0, 1)
    param_idx1_range = (0, 2*np.pi, 0.1)   # Lower limit, upper limit, step size
    param_idx2_range = (0, 2*np.pi, 0.1)   # Lower limit, upper limit, step size

    # qnn
    x_wires = list(range(x_qbits))
    qnn = get_qnn(qnn_name, x_wires, num_layers, device='cpu')
    qnn.params = qnn.params.detach()
    qnn.params[0, 0, 2] = 0

    # Unitray U
    if U is None:
        # U, U_params = create_unitary_from_circuit(qnn_name, x_wires, num_layers, device=device)
        # qnn.params = U_params
        U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)


    # Dataset X
    schmidt_rank = 2**r
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    if X is None:
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
            if func is None:
                loss = cost_func(X, y_conj, qnn, device=device)
            else:
                loss = func(cost_func(X, y_conj, qnn, device=device))
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

    plotly.offline.plot(fig, filename=f'./plots/loss_map/{name}.html')
    # fig.show()


def map_loss_function(U=None, X=None, func=None, name='loss_map'):
    from classic_training import cost_func
    # Parameters
    x_qbits = 1
    r = 0
    num_points = 2
    qnn_name = 'CudaPennylane'
    device = 'cpu'
    num_layers = 1

    param_idx1 = (0, 0, 0)
    param_idx2 = (0, 0, 1)
    param_idx1_range = (0, 2*np.pi, 0.1)   # Lower limit, upper limit, step size
    param_idx2_range = (0, 2*np.pi, 0.1)   # Lower limit, upper limit, step size

    # qnn
    x_wires = list(range(x_qbits))
    qnn = get_qnn(qnn_name, x_wires, num_layers, device='cpu')
    qnn.params = qnn.params.detach()
    qnn.params[0, 0, 2] = 0


    # Unitray U
    if U is None:
        # U, U_params = create_unitary_from_circuit(qnn_name, x_wires, num_layers, device=device)
        # qnn.params = U_params
        U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128, device=device)


    # Dataset X
    schmidt_rank = 2**r
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    if X is None:
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
            if func is None:
                result[-1].append(cost_func(X, y_conj, qnn, device=device))
            else:
                result[-1].append(func(cost_func(X, y_conj, qnn, device=device)))

    import plotly.graph_objects as go
    import plotly

    result = np.array(result)

    fig = go.Figure(data=[go.Surface(x=param1_list, y=param2_list, z=result)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Loss Map', autosize=True)

    plotly.offline.plot(fig, filename=f'./plots/loss_map/{name}.html')
    # fig.show()


def power_func(power):
    def func(x):
        return torch.float_power(torch.abs(x), power)
    return func


def root_func(power):
    def func(x):
        return x-x if x < 1e-2 else torch.float_power(torch.abs(x), 1/power)
    return func


def funky_func():
    def func(x):
        return 1-(torch.exp(-x*10))
    return func


if __name__ == '__main__':

    # U = torch.tensor(random_unitary_matrix(1), dtype=torch.complex128, device='cpu')
    # qnn = get_qnn('CudaPennylane', [0], 1)
    # qnn.params = qnn.params.detach()
    # qnn.params[0, 0, 2] = 0
    # U = qnn.get_tensor_V()
    # map_loss_function(U=U)
    # gradient_3D_plot(U=U)
    # map_loss_function(U=U, func=root_func(2), name='loss_map_squared')
    # gradient_3D_plot(U=U, func=root_func(2), name='gradient_map_squared')
    # # map_loss_function(U=U, func=root_func(3), name='loss_map_pow_3')
    # # gradient_3D_plot(U=U, func=root_func(3), name='gradient_map_pow_3')
    # # map_loss_function(U=U, func=root_func(4), name='loss_map_pow_4')
    # # gradient_3D_plot(U=U, func=root_func(4), name='gradient_map_pow_4')
    # # map_loss_function(U=U, func=root_func(5), name='loss_map_pow_5')
    # # gradient_3D_plot(U=U, func=root_func(5), name='gradient_map_pow_5')
    # map_loss_function(U=U, func=root_func(7), name='loss_map_pow_7')
    # gradient_3D_plot(U=U, func=root_func(7), name='gradient_map_pow_7')
    # map_loss_function(U=U, func=root_func(100), name='loss_map_pow_100')
    # gradient_3D_plot(U=U, func=root_func(100), name='gradient_map_pow_100')
    scheduler_factor = 0.8
    scheduler_patience = 10
    num_processes = 8
    lr = 0.1
    run_type = 'new'
    schmidt_ranks = [4]
    num_datapoints = None
    cost_modification = funky_func
    exp(4, 60, 1000, lr, 10, 100, 'CudaPennylane', 'cpu', None, True, 'Adam',
        scheduler_factor=scheduler_factor, scheduler_patience=scheduler_patience, std=True,
        writer_path='./experimental_results/enhanced_plateaus/', num_processes=num_processes, run_type=run_type,
        small_std=False,
        schmidt_ranks=schmidt_ranks,
        num_datapoints=num_datapoints,
        cost_modification=cost_modification
        )
    # exp(4, 45, 1000, 0.1, 1, 1, 'CudaPennylane', 'cpu', None, True, 'Adam', std=True,
    #     writer=Writer('./experimental_results/4_qubit_exp_45_std.txt'))
    #exp(x_qbits, num_layers, num_epochs, lr, num_unitaries, num_datasets, qnn_name, device, cheat, use_scheduler,
        #optimizer, scheduler_factor=0.8, scheduler_patience=3, std=False, writer_path=None):
    # schmidt_rank = 1
    # r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    # x_qbits = 1
    # size = 2
    # U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128)
    # X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, size, x_qbits, r_qbits)))
    # gradient_3D_plot(U, X, func=torch.log, name='nat_log')
    # gradient_3D_plot(U, X, func=root_func(5), name='root5')
    # gradient_3D_plot(U, X, func=root_func(11), name='root11')
    # gradient_3D_plot(U, X, name='normal')
    # map_loss_function(U, X, func=funky_func(), name='funky')
