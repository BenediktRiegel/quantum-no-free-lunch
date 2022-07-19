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
