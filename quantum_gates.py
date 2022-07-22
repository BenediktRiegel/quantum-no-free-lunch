import torch
import numpy as np

small_I = None
other_dig_j = None
other_dig_one_and_minus_one = None
one_top_left = None
one_top_left = None
one_bottom_right = None
_H = None
_CNOT = None


def init_globals(device='cpu'):
    global small_I
    global other_dig_j
    global other_dig_one_and_minus_one
    global one_top_left
    global one_top_left
    global one_bottom_right
    global _H
    global _CNOT
    small_I = torch.tensor([
        [1, 0],
        [0, 1]
    ], dtype=torch.complex128, device=device)

    other_dig_j = torch.tensor([
        [0, 1j],
        [1j, 0]
    ], dtype=torch.complex128, device=device)

    other_dig_one_and_minus_one = torch.tensor([
        [0, -1],
        [1, 0]
    ], dtype=torch.complex128, device=device)

    one_top_left = torch.tensor([
        [1, 0],
        [0, 0]
    ], dtype=torch.complex128, device=device)

    one_bottom_right = torch.tensor([
        [0, 0],
        [0, 1]
    ], dtype=torch.complex128, device=device)

    _H = torch.tensor([
        [1. / np.sqrt(2.), 1. / np.sqrt(2.)],
        [1. / np.sqrt(2.), -1. / np.sqrt(2.)]
    ], dtype=torch.complex128, device=device)

    _CNOT = torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=torch.complex128, device=device)


init_globals(device='cpu')


def I(size=2, device='cpu'):
    return torch.eye(size, device=device)


def CNOT():
    return _CNOT


def RX(rx):
    x_sin = torch.sin(rx / 2.)
    x_cos = torch.cos(rx / 2.)
    result = small_I*x_cos - other_dig_j*x_sin
    return result


def RY(ry):
    y_sin = torch.sin(ry / 2.)
    y_cos = torch.cos(ry / 2.)
    result = small_I*y_cos + other_dig_one_and_minus_one*y_sin
    return result


def RZ(rz):
    z_exp = torch.exp(1j*rz)
    return one_top_left + z_exp*one_bottom_right


def U3(rx, ry, rz):
    return torch.matmul(torch.matmul(RZ(rz), RY(ry)), RX(rx))


def H():
    return H


def is_unitary(M, error=1e-15):
    zeros = (M @ M.T.conj()) - I(size=M.shape[0])
    return ((zeros.real < error).all and (zeros.imag < error).all()).item()


def main():
    for _ in range(40):
        r = torch.from_numpy(np.random.normal(0, 2*np.pi, (3,)))
        # r = torch.tensor([0, 0, 0])
        print('RX, RY, RZ, U3')
        print(f'{is_unitary(RX(r[0]))}, {is_unitary(RY(r[1]))}, {is_unitary(RZ(r[2]))}, {is_unitary(U3(r[0], r[1], r[2]))}')


if __name__ == '__main__':
    main()
