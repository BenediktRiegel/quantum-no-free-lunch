from config import gen_config
from test import calc_avg_risk
import json
import numpy as np
from log import Logger
from os.path import exists


class Writer:
    def __init__(self, file_path):
        self.file_path = file_path
        f = open(file_path, 'w')
        f.write('')
        f.close()

    def append(self, text):
        text = str(text)
        with open(self.file_path, 'a') as f:
            f.write(text + '\t')
            f.close()


def load_txt_results(save_dir: str):
    result_path = save_dir + 'result.txt'
    config_path = save_dir + 'config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
        f.close()

    txt_results = []
    with open(result_path, 'r') as f:
        txt_results = f.readline().split('\t')
        f.close()

    all_risks = []
    idx = 0
    for i in range(config['x_qbits'] + 1):
        if idx >= len(txt_results):
            pass
        risk_list = []
        for num_points in range(1, config['num_points'] + 1):
            if idx >= len(txt_results):
                pass
            idx += 1
            risk_list.append(txt_results[idx])
        all_risks.append(risk_list)
    return np.array(all_risks)


def exp_fig2_3(config, save_dir):
    # store config
    with open(save_dir + 'config.json', 'w') as f:
        json.dump(config, f)
        f.close()

    logger = Logger(config['x_qbits'], config['num_points'], config['num_unitaries'],
                    config['num_training_data'])
    writer = Writer(save_dir + 'result.txt')
    all_risks = []
    for i in range(config['x_qbits'] + 1):
        rank = 2**i
        risk_list = []
        # print(f"rank {i+1}/{config['x_qbits'] + 1} (rank={rank})")
        r_qbits = i
        logger.update_schmidt_rank(i)
        for num_points in range(1, config['num_points'] + 1):
            logger.update_num_points(num_points)
            # print(f"num_points {num_points}/{config['num_points']}")
            risk = calc_avg_risk(rank, num_points, config['x_qbits'],
                                 r_qbits, config['num_unitaries'],
                                 config['num_layers'], config['num_training_data'],
                                 config['learning_rate'], config['num_epochs'],
                                 config['batch_size'],
                                 False, 0,
                                 logger
                                 )
            # Store risks directly
            writer.append(risk)
            risk_list.append(risk)
        all_risks.append(risk_list)
    all_risks_array = np.array(all_risks)

    # store risks
    np.save(save_dir + 'result.npy', all_risks_array)

def test_fig2():
    print("start experiment 1")
    #Fig.2 Paper Sharma et al.
    config = gen_config(2, 2, 1, 1, 10, 10, 10, 0)
    print("config generated")
    exp_fig2_3(config, './experimental_results/exp1/')
    # all_risks = []
    # for rank in [2 ** i for i in range(config['x_qbits'] + 1)]:
    #     risk_list = []
    #     for num_points in range(1, config['num_points'] + 1):
    #         risk = calc_avg_risk(rank, num_points, config['x_qbits'],
    #                              config['r_qbits'], config['num_unitaries'],
    #                              config['num_layers'], config['num_training_data'],
    #                              config['learning_rate'], config['num_epochs']
    #                              )
    #         risk_list.append(risk)
    #     all_risks.append(risk_list)
    # all_risks_array = np.array(all_risks)
    #
    # # store config
    # with open('./experimental_results/exp1/config.json', 'w') as f:
    #     json.dump(config, f)
    #     f.close()
    #
    # # store risks
    # np.save('./experimental_results/exp1/result.npy', all_risks_array)


def test_fig3():
    print("start experiment 2")
    #Fig.3 Paper Sharma et al.
    #init config
    config = gen_config(1, 1, 6, 6, 10, 10, 100, 1, 0, 0.01, 8, 120, True, 'SGD')
    print("config generated")

    # small test version
    # config = gen_config(1,2,2,2,3,2,2,1,0,0.01,8,3,True)
    exp_fig2_3(config, './experimental_results/exp2/')
    #gen combis of number training samples and rank
    # all_risks = []
    # for rank in [2**i for i in range(config['num_qbits'] + 1)]:
    #     risk_list = []
    #     for std in range(0, upper_std):
    #         risk = calc_avg_risk(rank, num_points, config['x_qbits'],
    #                              config['r_qbits'], config['num_unitaries'], config['num_layers'],
    #                              config['num_training_data'])
    #         risk_list.append(risk)
    #     all_risks.append(risk_list)
    # all_risks_array = np.array(all_risks)
    #
    #
    # #store config
    # with open('./experimental_results/exp2/config.json', 'w') as f:
    #     json.dump(config, f)
    #     f.close()
    #
    # #store dict
    # np.save('./experimental_results/exp2/result.npy', all_risks_array)


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

    logger = Logger(config['x_qbits'], config['num_points'], config['num_unitaries'],
                    config['num_training_data'])
    writer = Writer(save_dir + 'result.txt')
    all_risks = []

    for std in range(0, upper_std):
        #logger.update_num_points(num_points)
        # print(f"num_points {num_points}/{config['num_points']}")
        risk = calc_avg_risk(config['rank'], config['num_points'], config['x_qbits'],
                                config['r_qbits'], config['num_unitaries'],
                                config['num_layers'], config['num_training_data'],
                                config['learning_rate'], config['num_epochs'],
                                config['batch_size'],
                                True, std, logger)


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

    logger = Logger(config['x_qbits'], config['num_points'], config['num_unitaries'],
                    config['num_training_data'])
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
            risk = calc_avg_risk(config['rank'], config['num_points'], config['x_qbits'],
                                 config['r_qbits'], config['num_unitaries'],
                                 config['num_layers'], config['num_training_data'],
                                 config['learning_rate'], config['num_epochs'],
                                 config['batch_size'],
                                 True, std,
                                 logger
                                 )
            # Store risks directly
            writer.append(risk)
            risk_list.append(risk)
        all_risks.append(risk_list)
    all_risks_array = np.array(all_risks)

    # store risks
    np.save(save_dir + 'result.npy', all_risks_array)




def test():

    test_dict = dict(
        test1='Frank',
        nummer=1,
    )

    # This saves a dict as json
    with open('./experimental_results/test.json', 'w') as f:
        json.dump(test_dict, f)
        f.close()
    # This loads a dict from json
    with open('./experimental_results/test.json', 'r') as f:
        result = json.load(f)
        f.close()


def main():
    test_fig2()
    # test_fig3()
    # two_points_rank_one()
    # print(np.load('./experimental_results/exp1/result.npy'))


if __name__ == '__main__':
    main()
