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
    exp_fig2_3(config, './experimental_results/exp1/')

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
    exp_fig2_3(config, './experimental_results/exp2/')


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








