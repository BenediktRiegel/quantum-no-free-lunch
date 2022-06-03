import pennylane as qml
import numpy as np
from scipy.stats import unitary_group


def hadamard_layer(wires):
    for i in wires:
        qml.Hadamard(i)


def schmidt_rank(data_wires, reference_wires, state):
    pass


# generates binary representation with num_bits for given number
def int_to_bin(num, num_bits):
    b = bin(num)[2:]
    return [0 for _ in range(num_bits - len(b))] + [int(el) for el in b]


def uniformly_sample_from_base(num_qbits, size):
    #uniform sampling of basis vectors
    base = []
    random_ints = np.random.choice(np.power(2, num_qbits), size, replace=False)
    for rd_int in range(len(random_ints)):
        binary_base = int_to_bin(random_ints[rd_int], num_qbits)
        base.append(binary_base)
    return base


def normalize(point):
    return point / np.linalg.norm(point)


# Return a randomly uniformly sampled point with schmidt rank <schmid_rank> and size x_qbits+r_qbits
def uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits):
    basis_x = uniformly_sample_from_base(x_qbits, schmidt_rank)
    basis_r = uniformly_sample_from_base(r_qbits, schmidt_rank)
    coeff = np.random.uniform(size=schmidt_rank)
    point = np.zeros((x_qbits + r_qbits))
    for i in range(schmidt_rank):
        point += coeff[i] * np.array(basis_x[i] + basis_r[i])
    return normalize(point)

#create dataset of size <size> with a given schmidt rank
def uniform_random_data(schmidt_rank, size, x_qbits, r_qbits):
    data = []
    #size = number data samples of trainset
    for i in range(size):
        data.append(uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits))
    return data



#helper function to generate random samples of a certain mean value
# params: mean of generate numbers with standard deviation of certain value
#         number of samples to generate
def create_mean_std(mean, std, num_samples):
    data = []
    samples = np.random.normal(loc=0.0, scale=std, size=num_samples)
    print('testing of mean_std data generation')
    actual_mean = np.mean(samples)
    actual_std = np.std(samples)
    print("Initial samples stats   : mean = {:.4f} stdv = {:.4f}".format(actual_mean, actual_std))
    zero_mean_samples = samples - (actual_mean)

    zero_mean_mean = np.mean(zero_mean_samples)
    zero_mean_std = np.std(zero_mean_samples)
    print("True zero samples stats : mean = {:.4f} stdv = {:.4f}".format(zero_mean_mean, zero_mean_std))

    scaled_samples = zero_mean_samples * (std/zero_mean_std)
    scaled_mean = np.mean(scaled_samples)
    scaled_std = np.std(scaled_samples)
    print("Scaled samples stats    : mean = {:.4f} stdv = {:.4f}".format(scaled_mean, scaled_std))

    final_samples = scaled_samples + mean
    final_mean = np.mean(final_samples)
    final_std = np.std(final_samples)
    print("Final samples stats     : mean = {:.4f} stdv = {:.4f}".format(final_mean, final_std))


    return final_samples


#create dataset of size <size> with a mean schmidt rank
def uniform_random_data_mean(mean, std, num_samples,size, x_qbits, r_qbits):
    data = []
    numbers_mean_std= create_mean_std(mean,std, num_samples)
    for i in range(len(numbers_mean_std)):
        schmidt_rank = numbers_mean_std[i]
        data.append(uniformly_sample_random_point(schmidt_rank, x_qbits, r_qbits))
    return data


def random_unitary_matrix(x_qbits):
    matrix = unitary_group.rvs(x_qbits)
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


def test_unitary():
    dev = qml.device('default.qubit', wires=[3])

    unitary = random_unitary_matrix(np.power(2, 3))

    @qml.qnode(dev)
    def circuit():
        unitary_circuit(unitary)(wires=range(3))

        return [qml.expval(qml.PauliZ((m,))) for m in range(3)]

    print(qml.draw(circuit)())


if __name__ == '__main__':
    test_unitary()
    # create_mean_std(10, 4, 100)
