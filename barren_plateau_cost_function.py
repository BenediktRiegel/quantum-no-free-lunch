import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from data import *
import torch

def cost_func(X_train, unitary, ref_wires, dev, random_gate_sequence, params):
    #input params: train data, qnn, unitary to learn, refernce system wires and device
    qnn_wires = list(range(0,num_qubits))
    cost = torch.zeros(1)
    for el in X_train:
        @qml.qnode(dev, interface="torch")
        def circuit(params):
            qml.QubitStateVector(el, wires=qnn_wires+ref_wires)  # Amplitude Encoding
            #estimate V
            # construct random circuit for V
            for i in range(num_qubits):
                qml.RY(np.pi / 4, wires=i)
            for i in range(num_qubits):
                random_gate_sequence[i](params[i], wires=i)
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i + 1])
            adjoint_unitary_circuit(unitary)(wires=qnn_wires)  # Adjoint U
            qml.MottonenStatePreparation(el, wires=qnn_wires+ref_wires).inv()  # Inverse Amplitude Encoding
            return qml.probs(wires=qnn_wires+ref_wires)
        cost += circuit(params)[0]

    return 1 - (cost / len(X_train))




def get_gradient(unitary, learning_rate, ref_wires,
              dev, num_epochs, random_gate_sequence):
    # num_qubits = len(qnn.wires) + len(ref_wires)
    # num_layers = qnn.num_layers
    # set up the optimizer
    opt = torch.optim.Adam([params], lr=learning_rate)
    # opt = torch.optim.SGD([qnn.params], lr=learning_rate)

    # number of steps in the optimization routine
    steps = num_epochs

    # the final stage of optimization isn't always the best, so we keep track of
    # the best parameters along the way
    # best_cost = 0
    # best_params = np.zeros((num_qubits, num_layers, 3))

    # optimization begins
    all_losses = []
    gradient_storage = []
    for n in range(1):
        print(f"step {n+1}/{steps}")
        opt.zero_grad()
        total_loss = 0
        for el in X_train:
            print('calc cost funktion')
            loss = cost_func(el, unitary, ref_wires, dev, random_gate_sequence, params)
            print('backprop')
            loss.backward()
            gradient_storage.append(params.grad)
            print('optimise')
            opt.step()



    return np.mean(gradient)

num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)
gate_set = [qml.RX, qml.RY, qml.RZ]
grad_vals = []
num_samples = 200
schmidt_rank = 1
num_points = 64
grads = []
for i in range(num_samples):
    gate_sequence = {i: np.random.choice(gate_set) for i in range(num_qubits)}
    params = np.random.uniform(0, 2*np.pi, size=num_qubits)
    params = Variable(torch.tensor(params), requires_grad=True)
    unitary = random_unitary_matrix(num_qubits)
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    ref_wires = list(range(num_qubits,num_qubits+r_qbits))
    X_train = np.array(uniform_random_data(schmidt_rank, num_points, num_qubits, r_qbits))
    #print(rand_circuit(X_train, unitary, schmidt_rank, params, gate_sequence, num_qubits))
    gradient = get_gradient(unitary,0.1, ref_wires,dev, gate_sequence, params)
    grad_vals.append(gradient)


print("Variance of the gradients for {} random circuits: {}".format(
    num_samples, np.var(grad_vals)
)
)
print("Mean of the gradients for {} random circuits: {}".format(
    num_samples, np.mean(grad_vals)
)
)

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


"""
def rand_circuit(X_train, unitary, schmidt_rank, params, random_gate_sequence=None, num_qubits=None):
    cost = 0
    qnn_wires = list(range(0,num_qubits))
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    ref_wires = list(range(num_qubits,num_qubits+r_qbits))
    for el in X_train:
        @qml.qnode(dev)
        def circuit():
            qml.MottonenStatePreparation(el, wires=qnn_wires + ref_wires)  # Amplitude Encoding
            #construct random circuit for V
            for i in range(num_qubits):
                qml.RY(np.pi / 4, wires=i)
            for i in range(num_qubits):
                random_gate_sequence[i](params[i], wires=i)
            for i in range(num_qubits - 1):
                qml.CZ(wires=[i, i + 1])

            adjoint_unitary_circuit(unitary)(wires=qnn_wires)  # Adjoint U
            qml.MottonenStatePreparation(el, wires=qnn_wires + ref_wires).inv()  # Inverse Amplitude Encoding
            #H = np.zeros((2 ** num_qubits, 2 ** num_qubits))
            #H[0, 0] = 1
            #wirelist = [i for i in range(num_qubits)]
            #return qml.expval(qml.Hermitian(H, wirelist))
            return qml.probs(wires=qnn_wires + ref_wires)  # Return probabilities for differen quantum states

        cost +=  circuit()[0]  # Sum up probability of state |0>

    return 1 - (cost / len(X_train))


num_qubits = 4
dev = qml.device("default.qubit", wires=num_qubits)
gate_set = [qml.RX, qml.RY, qml.RZ]
grad_vals = []
num_samples = 200
schmidt_rank = 1
num_points = 64

for i in range(num_samples):
    gate_sequence = {i: np.random.choice(gate_set) for i in range(num_qubits)}
    qcircuit = qml.QNode(rand_circuit, dev)
    #grad = qml.grad(qcircuit, argnum=0)
    params = np.random.uniform(0, 2*np.pi, size=num_qubits)
    unitary = random_unitary_matrix(num_qubits)
    r_qbits = int(np.ceil(np.log2(schmidt_rank)))
    X_train = np.array(uniform_random_data(schmidt_rank, num_points, num_qubits, r_qbits))
    print(rand_circuit(X_train, unitary, schmidt_rank, params, gate_sequence, num_qubits))

    #params are: X_train, unitary, schmidt_rank, params, random_gate_sequence=None, num_qubits=None
    #gradient = grad(X_train, unitary, schmidt_rank, params, random_gate_sequence=gate_sequence, num_qubits=num_qubits)
    #grad_vals.append(gradient[-1])

print("Variance of the gradients for {} random circuits: {}".format(
    num_samples, np.var(grad_vals)
)
)
print("Mean of the gradients for {} random circuits: {}".format(
    num_samples, np.mean(grad_vals)
)
)

qubits = [2, 3, 4, 5, 6]
variances = []

for num_qubits in qubits:
    grad_vals = []
    for i in range(num_samples):
        dev = qml.device("default.qubit", wires=num_qubits)
        qcircuit = qml.QNode(rand_circuit, dev)
        grad = qml.grad(qcircuit, argnum=0)

        gate_set = [qml.RX, qml.RY, qml.RZ]
        random_gate_sequence = {i: np.random.choice(gate_set) for i in range(num_qubits)}

        params = np.random.uniform(0, np.pi, size=num_qubits)
        gradient = grad(
            params, random_gate_sequence=random_gate_sequence, num_qubits=num_qubits
        )
        grad_vals.append(gradient[-1])
    variances.append(np.var(grad_vals))

variances = np.array(variances)
qubits = np.array(qubits)

# Fit the semilog plot to a straight line
p = np.polyfit(qubits, np.log(variances), 1)

# Plot the straight line fit to the semilog
plt.semilogy(qubits, variances, "o")
plt.semilogy(qubits, np.exp(p[0] * qubits + p[1]), "o-.", label="Slope {:3.2f}".format(p[0]))
plt.xlabel(r"N Qubits")
plt.ylabel(r"Variance")
plt.legend()
plt.show()
"""




