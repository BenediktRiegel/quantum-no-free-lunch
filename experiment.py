from config import gen_config
from test import calc_avg_risk
import json
import numpy as np

def test_fig2():

    num_layers = 10
    #Fig.2 Paper Sharma et al.
    # Unentangled training
    dict_result = {}
    config = gen_config(2, 2, 1, 1, 10, 10, 10, 0)
    for num_points in range(1, config['num_points'] + 1):
        risks_for_current_rank = []
        for rank in [2 ** i for i in range(config['x_qbits'] + 1)]:
            risk = calc_avg_risk(rank, num_points, config['x_qbits'],
                                 config['r_qbits'], config['num_unitaries'],
                                 config['num_layers'], config['num_training_data'],
                                 config['learning_rate'], config['num_epochs']
                                 )
            dict_result[str(num_points)][str(rank)] = risk

    # store config
    with open('./experimental_results/exp1/config.json', 'w') as f:
        json.dump(config, f)
        f.close()

    # store dict
    with open('./experimental_results/exp1/result.json', 'w') as f:
        json.dump(dict_result, f)
        f.close()


def test_fig3():
    #Fig.3 Paper Sharma et al.
    #init config

    dict_result = dict()
    #config =get_config(1,1,6,6, num_unitaries=10, 10, 100,1,0,0.01,8,120,True)

    # small test version
    config = gen_config(1,2,2,2,3,2,2,1,0,0.01,8,3,True)
    #gen combis of number training samples and rank
    all_risks = []
    for rank in [2**i for i in range(config['num_qbits'] + 1)]:
        risk_list = []
        for num_points in range(1, config['num_points'] + 1):
            risk = calc_avg_risk(rank, num_points, config['x_qbits'],
                                 config['r_qbits'], config['num_unitaries'], config['num_layers'],
                                 config['num_training_data'])
            risk_list.append(risk)
        all_risks.append(risk_list)
    all_risks_array = np.array(all_risks)


    #store config
    with open('./experimental_results/exp2/config.json', 'w') as f:
        json.dump(config, f)
        f.close()

    #store dict
    np.save(all_risks_array,'./experimental_results/exp2/result.npy')


def test():

    test_dict = dict(
        test1='Frank',
        type='Opfer',
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


if __name__ == '__main__':
    main()
