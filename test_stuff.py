import pennylane as qml
import torch


def init_entanglement_layers():
    num_wires = 6
    ent_layers = []

    def ent_layer():
        if num_wires > 1:
            for i in range(num_wires):
                c_wire = i
                t_wire = (i + 1) % num_wires
                qml.CNOT(wires=[c_wire, t_wire])

    return torch.tensor(qml.matrix(ent_layer)(), dtype=torch.complex128)


def prep_rot_matrix():
    num_wires = 6
    matrix = "torch.stack(["
    for p0 in range(4):
        for p1 in range(4):
            for p2 in range(4):
                for p3 in range(4):
                    for p4 in range(4):
                        for p5 in range(4):
                            matrix += f"r0[{p0}]*r1[{p1}]*r2[{p2}]*r3[{p3}]*r4[{p4}]*r5[{p5}], "

    matrix = matrix[:-2]
    matrix += "]).reshape(64, 64)"
    return matrix


def main():
    import quantum_gates as qg
    import pennylane as qml
    # U = qg.H()
    U = torch.tensor([
        [2, 3],
        [4, 5]
    ])
    c_wire = 1
    CU = qg.controlled_U(c_wire, 0, U)
    print(CU)
    dev = qml.device('default.qubit', wires=c_wire+1)
    @qml.qnode(dev)
    def circuit():
        qml.ControlledQubitUnitary(U, control_wires=c_wire, wires=0)
        return qml.sample(wires=range(c_wire+1))
    truth = qml.matrix(circuit)()
    print(truth)
    print(qml.draw(circuit)())
    diff = (truth - CU)
    print(((diff.real < 1e-15).all() and (diff.imag < 1e-15).all()).item())


if __name__ == '__main__':
    main()
