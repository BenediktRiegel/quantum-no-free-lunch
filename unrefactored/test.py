from qnn import *
from utils import quantum_risk
import matplotlib.pyplot as plt
import time
from log import Logger

"""
def construct_circuit(trained_params, num_layers, x_qbits):
    qnn = PennylaneQNN(wires=list(range(x_qbits)), num_layers=num_layers)
    return qnn.qnn()
"""


def calc_risk_qnn(trained_qnn, U):

    #circuit = construct_circuit(trained_params, num_layers, x_qbits)
    V = trained_qnn.get_matrix_V()
    risk = quantum_risk(U, V)
    return risk


def calc_avg_std_risk(schmidt_rank, num_points, x_qbits, r_qbits,
                  num_unitaries, num_layers, num_training_data,
                  learning_rate, num_epochs, batch_size, mean_std, std, logger: Logger,
                  qnn=None):
    all_risks = []
    for i in range(num_unitaries):
        # Draw random unitary
        logger.update_num_unitary(i)
        unitary = random_unitary_matrix(x_qbits)
        # Train it with <num_train_data> many random datasets
        for j in range(num_training_data):
            logger.update_num_training_dataset(j)
            # Init data and neural net
            if mean_std:
                dataset = SchmidtDataset_std(schmidt_rank, num_points, x_qbits, r_qbits, std)
            else:
                dataset = SchmidtDataset(schmidt_rank, num_points, x_qbits, r_qbits)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            qnn = PennylaneQNN(wires=list(range(x_qbits)), num_layers=num_layers)
            # Init quantum device
            ref_wires = list(range(x_qbits, x_qbits+r_qbits))
            
            dev = qml.device('lightning.qubit', wires=qnn.wires+ref_wires)
            dev.shots = 1000
            # Train and compute risk
            print('training qnn')
            start_time = time.time()
            train_qnn(qnn, unitary, dataloader, ref_wires, dev, learning_rate, num_epochs)
            total_time = time.time() - start_time
            print(f"training took {total_time}s")
            # plt.plot(list(range(len(losses))), losses)
            print('calculating risk')
            risk = calc_risk_qnn(qnn, unitary)
            all_risks.append(risk)
    # plt.grid(True)
    # plt.show()

    #average over all unitaries and training sets
    print('average risk')
    all_risks = np.array(all_risks)
    return all_risks.mean(), all_risks.std()


def main():
    from config import gen_config
    gen_config()
    print(dict.values())
    print(dict.keys())
    # print(calc_avg_risk(**params))


if __name__ == '__main__':
    main()
