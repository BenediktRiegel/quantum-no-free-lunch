import pennylane as qml
import numpy as np
from scipy.stats import unitary_group
import torch
from scipy.optimize import fmin_cobyla
from scipy.stats import unitary_group


def abs2(x):
    return x.real**2 + x.imag**2


def hadamard_layer(wires):
    for i in wires:
        qml.Hadamard(i)


def schmidt_rank(data_wires, reference_wires, state):
    pass


# generates binary representation with num_bits for given number
def int_to_bin(num, num_bits):
    b = bin(num)[2:]
    return [0 for _ in range(num_bits - len(b))] + [int(el) for el in b]


def one_hot_encoding(num, num_bits):
    result = [0]*num_bits
    result[num] = 1
    return result


def uniformly_sample_from_base(num_qbits, size):
    # uniform sampling of basis vectors
    num_bits = np.power(2, num_qbits)
    base = []
    random_ints = np.random.choice(num_bits, size, replace=False)
    transform_matrix = unitary_group.rvs(num_bits)
    for rd_int in range(len(random_ints)):
        binary_base = one_hot_encoding(random_ints[rd_int], num_bits)
        base.append(binary_base)


    return np.array(base) @ transform_matrix


def normalize(point):
    return point / np.linalg.norm(point)


def tensor_product(state1: np.ndarray, state2: np.ndarray):
    result = np.zeros(len(state1)*len(state2), dtype=np.complex128)
    for i in range(len(state1)):
        result[i*len(state2):i*len(state2)+len(state2)] = state1[i] * state2
    return result


def torch_tensor_product(matrix1: torch.Tensor, matrix2: torch.Tensor, device='cpu'):
    result = torch.zeros((matrix1.shape[0]*matrix2.shape[0], matrix1.shape[1]*matrix2.shape[1]), dtype=torch.complex128, device=device)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            result[i*matrix2.shape[0]:i*matrix2.shape[0]+matrix2.shape[0], j*matrix2.shape[1]:j*matrix2.shape[1]+matrix2.shape[1]] = matrix1[i, j] * matrix2
    return result


# Return a randomly uniformly sampled point with schmidt rank <schmid_rank> and size x_qbits+r_qbits
def uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits):
    basis_x = uniformly_sample_from_base(x_qbits, schmidt_rank)
    basis_r = uniformly_sample_from_base(r_qbits, schmidt_rank)
    coeff = np.random.uniform(size=schmidt_rank)
    point = np.zeros((2**x_qbits * 2**r_qbits), dtype=np.complex128)
    for i in range(schmidt_rank):
        point += coeff[i] * tensor_product(basis_x[i], basis_r[i])
    return normalize(point)


# create dataset of size <size> with a given schmidt rank
def uniform_random_data(schmidt_rank, size, x_qbits, r_qbits):
    data = []
    # size = number data samples of trainset
    for i in range(size):
        data.append(uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits))
    return data


# helper function to generate random samples of a certain mean value
# params: mean of generated numbers with standard deviation of certain value
#         number of samples to generate
def create_mean_std(mean, std, num_samples, max_rank, counter):
    data = []

    min_dist = min(mean - 1, max_rank - mean)
    if(min_dist <= 3 * std):
        raise ValueError(f'Bad standard deviation')
    samples = np.random.normal(loc=0.0, scale= std, size=num_samples)

    #samples = np.random.randint(mean - 5 * std, mean + 5 * std, size=num_samples)

    print('testing of mean_std data generation')
    actual_mean = np.mean(samples)
    actual_std = np.std(samples)
    #print(samples)
    print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))
    zero_mean_samples = samples - (actual_mean)

    zero_mean_mean = np.mean(zero_mean_samples)
    zero_mean_std = np.std(zero_mean_samples)
    #print(zero_mean_samples)
    print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))

    scaled_samples = zero_mean_samples * (std/zero_mean_std)
    scaled_mean = np.mean(scaled_samples)
    scaled_std = np.std(scaled_samples)
    #print(scaled_samples)
    print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))

    final_samples = scaled_samples + mean
    final_samples = np.round_(final_samples)
    final_mean = np.mean(final_samples)
    final_std = np.std(final_samples)
    print(final_samples)
    print("Final samples stats     : mean = {:.4f} stdv = {:.4f}".format(final_mean, final_std))

    if any(number <= 0 or number > max_rank for number in final_samples):
        counter = counter + 1
        final_samples , counter = create_mean_std(mean, std, num_samples, max_rank, counter)

    return final_samples, counter


#create dataset of size <size> with a mean schmidt rank
def uniform_random_data_mean(mean, std, num_samples, x_qbits, r_qbits):
    data = []
    numbers_mean_std, counter = create_mean_std(mean,std, num_samples)
    for i in range(len(numbers_mean_std)):
        schmidt_rank = numbers_mean_std[i]
        data.append(uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits))
    return data


def random_unitary_matrix(x_qbits):
    matrix = unitary_group.rvs(2**x_qbits)
    return matrix

#testing purposes of Schimdt rank
#calculate matrix out of quantum state
def state_matrix(state):
    return np.outer(state, state)

def calc_Schmidt(state):
    matrix = state_matrix(state)
    s = np.linalg.svd(matrix, full_matrices=True, compute_uv=False)
    non_zero_sv = 0
    for i in range(len(s)):
        if s[i] != 0:
            non_zero_sv += 1

    return non_zero_sv


def check_schmidt(state, schmidt_rank):
    return calc_Schmidt(state) == schmidt_rank


def main():
    schmidt_rank = 1
    data = uniform_random_data(schmidt_rank, 2, 1, 1)
    for state in data:
        try:
            if calc_Schmidt(state) != schmidt_rank:
                print(state)
        except:
            pass


def unitary_circuit(unitary):
    from qiskit import QuantumCircuit, Aer, transpile

    qbits = int(np.log2(len(unitary)))
    sv_backend = Aer.get_backend('statevector_simulator')

    qc = QuantumCircuit(qbits)
    qc.unitary(unitary, range(qbits))
    qc_transpiled = transpile(qc, backend=sv_backend, basis_gates=sv_backend.configuration().basis_gates,
                              optimization_level=3)
    return qml.from_qiskit(qc_transpiled)


def adjoint_unitary_circuit(unitary):
    from qiskit import QuantumCircuit, Aer, transpile

    unitary = np.conj(np.array(unitary)).T

    qbits = int(np.log2(len(unitary)))
    sv_backend = Aer.get_backend('statevector_simulator')

    qc = QuantumCircuit(qbits)
    qc.unitary(unitary, range(qbits))
    qc_transpiled = transpile(qc, backend=sv_backend, basis_gates=sv_backend.configuration().basis_gates,
                              optimization_level=3)
    return qml.from_qiskit(qc_transpiled)


def test_unitary():
    dev = qml.device('default.qubit', wires=[3])

    unitary = random_unitary_matrix(np.power(2, 3))

    @qml.qnode(dev)
    def circuit():
        unitary_circuit(unitary)(wires=range(3))

        return [qml.expval(qml.PauliZ((m,))) for m in range(3)]

    print(qml.draw(circuit)())


def quantum_risk(U, V): # <- signatur eventuell anpassen
    dim = len(U)
    U = np.matrix(U)
    V = np.matrix(V)
    prod = np.matmul(U.getH(), V)
    tr = abs(np.trace(prod))**2
    risk = 1 - ((dim + tr)/(dim * (dim+1)))
    
    return risk


def get_optimizer(optimizer_name: str, qnn_params, learning_rate):
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "cobyla":
        raise ValueError(f"Optimizer cobyla is not implemented!")
        fmin_cobyla()
    elif optimizer_name == 'spsa':
        raise ValueError(f"Optimizer spsa is not implemented!")
    elif optimizer_name == 'sgd':
        return torch.optim.SGD([qnn_params], lr=learning_rate)
    elif optimizer_name == 'adam':
        return torch.optim.Adam([qnn_params], lr=learning_rate)
    elif optimizer_name == 'nelder_mead':
        raise ValueError(f"Optimizer nelder_mead is not implemented!")
    raise ValueError(f"Optimizer {optimizer_name} is not implemented!")


def test_data_gen():
    uniform_random_data(2, 1, 3, 2)


if __name__ == '__main__':
    #test_data_gen()
    final_samples, counter =create_mean_std(15, 5, 100, 64,1)
    print("# of iterations:",  counter)
