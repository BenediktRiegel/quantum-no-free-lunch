from qnn import PennylaneQNN, cost_func, cost_func2, cost_func3
import pennylane as qml
from quantum_backends import QuantumBackends
from utils import uniform_random_data, random_unitary_matrix, transpile_adjoint_unitary, adjoint_unitary_circuit
import time


def measure_cost_func_time(num_layers, num_qbits):
    x_qbits = num_qbits
    r_qbits = num_qbits
    x_wires = list(range(num_qbits))
    r_wires = list(range(num_qbits, 2*num_qbits))
    qnn = PennylaneQNN(wires=x_wires, num_layers=num_layers)

    dev = QuantumBackends.aer_statevector_simulator.get_pennylane_backend('', '', qubit_cnt=2*num_qbits)
    dev.shots = 1024

    X = uniform_random_data(num_qbits, 1, x_qbits, r_qbits)
    U = random_unitary_matrix(x_qbits)
    # starting_time = time.time()
    # cost = cost_func(X, qnn, U, r_wires, dev)
    # total_time = time.time() - starting_time
    # print(f"The cost_func took {total_time}s")
    # print(f"cost = {cost}")
    # starting_time = time.time()
    # cost = cost_func2(X, qnn, U, r_wires, dev)
    # total_time = time.time() - starting_time
    # print(f"The cost_func2 took {total_time}s")
    # print(f"cost = {cost}")
    transpiled_U = adjoint_unitary_circuit(U)

    dev = QuantumBackends.qml_lightning.get_pennylane_backend('', '', qubit_cnt=2 * num_qbits)
    dev.shots = 1
    starting_time = time.time()
    cost = cost_func3(X, qnn, r_wires, dev, transpiled_U)
    total_time = time.time() - starting_time
    print(f"The cost_func3 took {total_time}s")
    print(f"cost = {cost}")
    return dev


def measure_hermitian_time(num_qbits):
    from qnn import get_density_matrix

    x_qbits = num_qbits
    r_qbits = num_qbits
    wires = list(range(num_qbits, 2*num_qbits))
    el = uniform_random_data(num_qbits, 1, x_qbits, r_qbits)[0]
    print('Start qml.Hermitian')
    starting_time = time.time()
    qml.Hermitian(get_density_matrix(el), wires=wires)
    total_time = time.time() - starting_time
    print(f"The circuit took {total_time}s")


def exp_duration_parameter_shift(x_qbits, num_layers, circuit_time, num_epochs=200, datasets=100, unitaries=10):
    num_parameters = 3*x_qbits*num_layers
    parameter_shift_duration = 2*num_parameters*circuit_time
    training_duration = parameter_shift_duration*num_epochs
    num_experiments = (x_qbits+1) * unitaries * datasets    # (x_qbits+1)=schmidt rank, 10=unitaries, 100=datasets
    experiment_duration = training_duration * num_experiments
    in_days = ((experiment_duration / 60.) / 60.) / 24.
    print(f"The experiment will take {in_days}d")


def main():
    measure_cost_func_time(10, 6)
    # measure_hermitian_time(6)
    # exp_duration_parameter_shift(6, 4, 0.15, datasets=50)


if __name__ == '__main__':
    main()
