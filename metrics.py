from qnn import *
from data import *
import classic_training
import quantum_training
import time
from logger import Writer


def quantum_risk(U, V):
    """
    Computes the quantum risk of a hypothesis unitary V
    with respect to the 'true' unitary U according to
    Equation A6 in Sharma et al.
    """
    dim = len(U)
    prod = torch.matmul(U.T.conj(), V)
    tr = abs(torch.trace(prod))**2
    risk = 1 - ((dim + tr)/(dim * (dim+1)))

    return risk


def calc_risk_qnn(trained_qnn, U):

    #circuit = construct_circuit(trained_params, num_layers, x_qbits)
    V = trained_qnn.get_matrix_V()
    risk = quantum_risk(U, V)
    return risk


def calc_avg_std_risk(schmidt_rank, num_points, x_qbits, r_qbits,
                  num_unitaries, num_layers, num_training_data,
                  learning_rate, num_epochs, batch_size, mean_std, std,
                  qnn=None, system_type='classic'):
    """
    classic/quantum
    """
    all_risks = []
    for i in range(num_unitaries):
        # Draw random unitary
        # logger.update_num_unitary(i)
        unitary = random_unitary_matrix(x_qbits)
        # Train it with <num_train_data> many random datasets
        for j in range(num_training_data):
            # logger.update_num_training_dataset(j)
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
            if(system_type == 'classic'):
                classic_training.train_qnn(qnn, unitary, dataloader, ref_wires, dev, learning_rate, num_epochs)
            elif(system_type == 'quantum'):
                quantum_training.train_qnn(qnn, unitary, dataloader, ref_wires, dev, learning_rate, num_epochs)
            
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

