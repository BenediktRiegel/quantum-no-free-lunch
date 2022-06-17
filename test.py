from qnn import *
import matplotlib.pyplot as plt

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


def calc_avg_risk(schmidt_rank, num_points, x_qbits, r_qbits,
                  num_unitaries, num_layers, num_training_data,
                  learning_rate, num_epochs):
    sum_risk = 0
    for i in range(num_unitaries):
        # Draw random unitary
        print(f"unitary {i+1}")
        unitary = random_unitary_matrix(x_qbits)
        # Train it with <num_train_data> many random datasets
        for j in range(num_training_data):
            print(f"training dataset {j + 1}")
            # Init data and neural net
            dataset = SchmidtDataset(schmidt_rank, num_points, x_qbits, r_qbits)
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            qnn = PennylaneQNN(wires=list(range(x_qbits)), num_layers=num_layers)
            # Init quantum device
            ref_wires = list(range(x_qbits, x_qbits+r_qbits))
            dev = qml.device('default.qubit', wires=qnn.wires+ref_wires)
            dev.shots = 1000
            # Train and compute risk
            losses = train_qnn(qnn, unitary, dataloader, ref_wires, dev, learning_rate, num_epochs)
            # plt.plot(list(range(len(losses))), losses)
            trained_params = qnn.params
            risk = calc_risk_qnn(qnn, unitary)
            sum_risk += risk
    # plt.grid(True)
    # plt.show()

    #average over all unitaries and training sets
    average_risk = sum_risk/(num_unitaries * num_training_data)
    return average_risk


def main():
    from config import gen_config
    gen_config()
    print(dict.values())
    print(dict.keys())
    # print(calc_avg_risk(**params))


if __name__ == '__main__':
    main()
