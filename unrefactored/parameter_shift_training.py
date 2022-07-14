from qnn import QNN, PennylaneQNN, cost_func, fast_cost_func
import pennylane as qml
from quantum_backends import QuantumBackends
from utils import uniform_random_data, random_unitary_matrix, adjoint_unitary_circuit
import time
import numpy as np
from typing import List
from utils import abs2, quantum_risk



np.random.seed(4241)


def shift_quantum_func(el, qnn: QNN, ref_wires: List[int], dev: qml.Device, transpiled_unitary, shift_idx, shift_factor):
    #input params: train data, qnn, unitary to learn, refernce system wires and device
    @qml.qnode(dev)
    def circuit():
        qml.QubitStateVector(el, wires=qnn.wires+ref_wires)  # Amplitude Encoding
        qnn.parameter_shift_qnn(shift_idx, shift_factor)
        transpiled_unitary(wires=qnn.wires)  # Adjoint U
        return qml.probs(wires=[0])
    # print('run circuit')
    circuit()
    # print('post processing')
    state = dev._state
    state = np.reshape(state, (2**(len(state.shape)),))
    prob = abs2(np.inner(el, state))

    return prob


def train(X, qnn, dev, transpiled_unitary, ref_wires, num_epochs, lr, decay=0.):
    for i in range(num_epochs):
        gradient_params = np.zeros(qnn.params.shape)
        for el in X:
            for idx in qnn.get_param_indices():
                plus = shift_quantum_func(el, qnn, ref_wires, dev, transpiled_unitary, idx, 1)
                minus = shift_quantum_func(el, qnn, ref_wires, dev, transpiled_unitary, idx, -1)
                gradient_params[idx] += minus - plus  # Parameter shift is plus - minus, but we get a minus from our cost function, which we apply here
        gradient_params /= len(X)

        # loss = fast_cost_func(X, qnn, ref_wires, dev, transpiled_unitary).item()
        print(f"epoch [{i+1}/{num_epochs}] loss={1}")

        if (gradient_params == 0.0).all():
            break
        qnn.params = qnn.params - lr * gradient_params
        lr *= (1. / (1. + decay * num_epochs))


def init(num_layers, num_qbits):
    x_qbits = num_qbits
    r_qbits = num_qbits
    x_wires = list(range(num_qbits))
    r_wires = list(range(num_qbits, 2*num_qbits))
    qnn = PennylaneQNN(wires=x_wires, num_layers=num_layers, use_torch=False)

    print('prep')
    X = uniform_random_data(1, 2, x_qbits, r_qbits)
    U = random_unitary_matrix(x_qbits)
    transpiled_U = adjoint_unitary_circuit(U)

    dev = QuantumBackends.qml_lightning.get_pennylane_backend('', '', qubit_cnt=2 * num_qbits)
    dev.shots = 1

    print('train')
    starting_time = time.time()
    train(X, qnn, dev, transpiled_U, r_wires, num_epochs=1, lr=0.1, decay=0.9)
    total_time = time.time() - starting_time
    print(f"training took {total_time}s")
    print(quantum_risk(U, qnn.get_matrix_V()))


def main():
    init(4, 6)


if __name__ == '__main__':
    main()
