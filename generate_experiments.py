from config import *
from  metrics import calc_avg_std_risk
import json
import numpy as np
from logger import Writer
from os.path import exists



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


def x_qubits_exp(x_qbits):
    import torch
    import time
    from data import random_unitary_matrix, uniform_random_data
    import importlib
    from classic_training import train
    from metrics import quantum_risk
    import matplotlib.pyplot as plt

    complete_starting_time = time.time()
    num_layers = 10
    num_epochs = 100
    lr = 0.1
    r_list = list(range(x_qbits+1))
    num_unitaries = 5
    num_datasets = 5
    num_datapoints = list(range(1, 2**x_qbits + 1))
    qnn_name = 'PennylaneQNN'

    results = dict()
    for r_idx in range(len(r_list)):
        schmidt_rank = 2**r_list[r_idx]
        results[r_list[r_idx]] = [1]
        for num_points_idx in range(len(num_datapoints)):
            num_points = num_datapoints[num_points_idx]
            risks = []
            for unitary_idx in range(num_unitaries):
                U = random_unitary_matrix(x_qbits)
                for dataset_idx in range(num_datasets):
                    print(f"Run: r [{r_idx+1}/{len(r_list)}], no. points [{num_points_idx+1}/{len(num_datapoints)}], "
                          f"U [{unitary_idx+1}/{num_unitaries}], dataset [{dataset_idx+1}/{num_datasets}]")

                    r_qbits = int(np.ceil(np.log2(schmidt_rank)))

                    x_wires = list(range(x_qbits))
                    qnn = getattr(importlib.import_module('qnn'), qnn_name)(wires=x_wires, num_layers=num_layers, use_torch=True)

                    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))

                    optimizer = torch.optim.Adam

                    if isinstance(qnn.params, list):
                        optimizer = optimizer(qnn.params, lr=lr)
                    else:
                        optimizer = optimizer([qnn.params], lr=lr)
                    scheduler = None
                    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.1)
                    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=10, min_lr=1e-10, verbose=True)
                    prep_time = time.time() - starting_time
                    print(f"\tPreparation with {x_qbits} qubits and {num_layers} layers took {prep_time}s")

                    starting_time = time.time()
                    losses = train(X, U, qnn, num_epochs, optimizer, scheduler)
                    train_time = time.time() - starting_time

                    risks.append(quantum_risk(U, qnn.get_matrix_V()))
            risks = np.array(risks)
            results[r_list[r_idx]].append(risks.mean())
    complete_total_time = time.time() - complete_starting_time
    print(f'experiment execution took {complete_total_time}s')

    for r_idx in range(len(r_list)):
        plt.plot([0] + num_datapoints, results[r_list[r_idx]], label=f'r={r_list[r_idx]}', marker='.')
    plt.xlabel('No. of Datapoints')
    plt.ylabel('Average Risk')
    plt.legend()
    plt.title('Average Risk for 3 Qubit Unitary')
    plt.tight_layout()
    plt.savefig(f'./plots/{x_qbits}_qubit_exp.png')
    plt.cla()


if __name__ == '__main__':
    x_qubits_exp(3)
