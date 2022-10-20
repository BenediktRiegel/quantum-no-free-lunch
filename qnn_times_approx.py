from classic_training import train
import torch
import numpy as np
from data import uniform_random_data, random_unitary_matrix
from qnns.qnn import get_qnn
from logger import Writer
import time
from metrics import quantum_risk
from concurrent.futures import ProcessPoolExecutor
from os.path import exists


def given_u_run(arg):
    (X, num_unitaries, x_qbits, layers, all_qnns, lr, scheduler_factor, num_epochs, writer, id) = arg
    for i in range(num_unitaries):
        U = torch.tensor(random_unitary_matrix(x_qbits), dtype=torch.complex128)
        for layer_idx in range(len(layers)):
            num_layers = layers[layer_idx]
            for qnn_idx in range(len(all_qnns)):
                qnn_name = all_qnns[qnn_idx]
                qnn = get_qnn(qnn_name, list(range(x_qbits)), num_layers)
                optimizer = torch.optim.Adam
                if isinstance(qnn.params, list):
                    optimizer = optimizer(qnn.params, lr=lr)
                else:
                    optimizer = optimizer([qnn.params], lr=lr)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=scheduler_factor, patience=3,
                                                                       min_lr=1e-10,
                                                                       verbose=False)
                print(f"ID={id}, qnn={qnn_name}, layers={num_layers}: "
                      f"U[{i+1}/{num_unitaries}], L[{layer_idx+1}/{len(layers)}], Q[{qnn_idx}/{len(all_qnns)}]")
                starting_time = time.time()
                losses = train(X, U, qnn, num_epochs, optimizer, scheduler)
                train_time = time.time() - starting_time
                risk = quantum_risk(U, qnn.get_matrix_V())
                writer.append_line(f"qnn={qnn_name}, num_layers={num_layers}, train_time={train_time}, risk={risk}, losses={losses}, unitary={U}")


def gen_qnn_vs_data(num_workers, num_unitaries, layers, writer, lr=0.1, scheduler_factor=0.85, num_epochs=1000, schmidt_rank=4, num_points=4, x_qbits=4):
    qqnns = [
        'PennylaneQNN', 'OffsetQNN', 'Circuit2QNN', 'Circuit5QNN', 'Circuit6QNN', 'Circuit9QNN',
        'Circuit11QNN', 'Circuit12QNN', 'Circuit13QNN', 'Circuit14QNN'
    ]
    cqnns = [
        'CudaPennylane', 'CudaCircuit6', 'CudaEfficient', 'CudaComplexPennylane', 'CudaSimpleEnt'
    ]
    all_qnns = qqnns + cqnns

    r_qbits = int(np.log2(schmidt_rank))
    X = torch.from_numpy(np.array(uniform_random_data(schmidt_rank, num_points, x_qbits, r_qbits)))
    X = X.reshape((X.shape[0], int(X.shape[1] / int(2**x_qbits)), int(2**x_qbits))).permute(0, 2, 1)
    if num_workers == 1:
        given_u_run((X, num_unitaries, x_qbits, layers, all_qnns, lr, scheduler_factor, num_epochs, writer, 0))
    else:
        ppe = ProcessPoolExecutor()
        inputs = [(X, num_unitaries, x_qbits, layers, all_qnns, lr, scheduler_factor, num_epochs, writer, i) for i in range(num_workers)]
        print('map to executor')
        ppe.map(given_u_run, inputs)
        print('done')


def main():
    num_unitaries = 10
    layers = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    uncomplete_file = './experimental_results/qnn_vs/qnn_vs_'
    file_path = uncomplete_file + '0.txt'
    file_num = 0
    while exists(file_path):
        file_num += 1
        file_path = uncomplete_file + f'{file_num}.txt'
    print(f'file_num = {file_num}')
    writer = Writer(file_path)
    num_workers = 1
    gen_qnn_vs_data(num_workers, num_unitaries, layers, writer)


if __name__ == '__main__':
    main()
