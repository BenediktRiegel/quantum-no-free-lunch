from os.path import exists
from logger import Writer


def gen_config(rank, num_points, x_qbits, r_qbits, num_unitaries, num_layers, num_training_data,
               mean, std = 0, learning_rate=0.01, batch_size=8, num_epochs=120, shuffle=True, optimizer='COBYLA'):
    '''
    schmidt_rank -- schmidt rank of states in training data (see param std)
    num_points -- number of points in each training dataset
    x_qbits -- number of input qubits
    r_qbits -- number of qubits in the reference system
    num_unitaries -- number of unitaries to be generated
    num_layers -- number of layers in NN architecture
    num_train_data -- number of training datasets
    std -- std deviation of schmidt rank (use schmidt_rank for all samples if this is 0)
    '''
    config = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        schmidt_rank=rank,
        num_points=num_points,
        x_qbits=x_qbits,
        r_qbits=r_qbits,
        num_unitaries=num_unitaries,
        num_layers=num_layers,
        num_training_data=num_training_data,
        mean=mean,
        std=std,
        optimizer=optimizer
    )
    return config


def get_exp_one_qubit_unitary_config():
    # Returns the config for the 1 qubit unitary experiment (fig2 in Sharma et al)
    return gen_config(2, 2, 1, 1, 10, 10, 10, 0)


def get_exp_six_qubit_unitary_config():
    # Returns the config for the 6 qubit unitary experiment (fig3 in Sharma et al)
    return gen_config(1, 1, 6, 6, 10, 10, 100, 1, 0, 0.01, 8, 120, True, 'SGD')


def gen_exp_file(x_qbits, num_unitaries, num_datasets, std_bool=False):
    file_path = f'./data/{x_qbits}_exp_file.txt'
    writer = Writer(file_path)
    r_list = list(range(x_qbits + 1))
    for r_idx in range(len(r_list)):
        schmidt_rank = 2**r_list[r_idx]
        num_datapoints = list(range(1, 2 ** x_qbits + 1))
        for num_points_idx in range(len(num_datapoints)):
            num_points = num_datapoints[num_points_idx]
            for unitary_idx in range(num_unitaries):
                for dataset_idx in range(num_datasets):
                    if not std_bool:
                        writer.append_line(f"schmidt_rank={schmidt_rank}, num_points={num_points}, std=0, "
                                           f"unitary_idx={unitary_idx}, dataset_idx={dataset_idx}")
                    else:
                        max_rank = 2 ** x_qbits
                        for std in range(1, max_rank):
                            if min(schmidt_rank - 1, max_rank - schmidt_rank) < 3 * std:
                                continue
                            writer.append_line(f"schmidt_rank={schmidt_rank}, num_points={num_points}, std={std}, "
                                               f"unitary_idx={unitary_idx}, dataset_idx={dataset_idx}")
    return file_path
