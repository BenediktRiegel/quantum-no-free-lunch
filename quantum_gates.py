import torch
import numpy as np


def init_globals(device='cpu'):
    pass


def torch_tensor(A, B, device='cpu'):
    B_rows = B.shape[0]
    B_cols = B.shape[1]
    size = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])
    result = torch.zeros(size, device=device, dtype=torch.complex128)
    row_result_idx = 0
    for row_A_idx in range(A.shape[0]):
        col_result_idx = 0
        for col_A_idx in range(A.shape[1]):
            result[row_result_idx:row_result_idx+B_rows, col_result_idx:col_result_idx+B_cols] = A[row_A_idx, col_A_idx] * B
            col_result_idx += B_cols
        row_result_idx += B_rows
    return result

small_I = torch.tensor([
    [1, 0],
    [0, 1]
], dtype=torch.complex128)

def I(size=2, device='cpu'):
    return torch.eye(size, device=device)


def CNOT(device='cpu'):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], device=device, dtype=torch.complex128)

other_dig_j = torch.tensor([
        [0, 1j],
        [1j, 0]
    ], dtype=torch.complex128)

def RX(rx):
    x_sin = torch.sin(rx / 2.)
    x_cos = torch.cos(rx / 2.)
    result = small_I*x_cos - other_dig_j*x_sin
    return result

other_dig_one_and_minus_one = torch.tensor([
        [0, -1],
        [1, 0]
    ], dtype=torch.complex128)

def RY(ry):
    y_sin = torch.sin(ry / 2.)
    y_cos = torch.cos(ry / 2.)
    result = small_I*y_cos + other_dig_one_and_minus_one*y_sin
    return result

one_top_left = torch.tensor([
        [1, 0],
        [0, 0]
    ], dtype=torch.complex128)

one_bottom_right = torch.tensor([
        [0, 0],
        [0, 1]
    ], dtype=torch.complex128)

def RZ(rz):
    z_exp = torch.exp(1j*rz)
    return one_top_left + z_exp*one_bottom_right


def U3(rx, ry, rz, device='cpu'):
    # x_sin = torch.sin(rx/2.)
    # x_cos = torch.cos(rx/2.)
    # ll = torch.exp(1j * ry)
    # ur = torch.exp(1j*rz)
    # return torch.tensor([
    #     [x_cos, -ur*x_sin],
    #     [ll * x_sin, ll*ur*x_cos]
    # ], device=device, dtype=torch.complex128)
    return torch.matmul(torch.matmul(RZ(rz), RY(ry)), RX(rx)).to(device)


_H = torch.tensor([
        [1./np.sqrt(2.), 1./np.sqrt(2.)],
        [1./np.sqrt(2.), -1./np.sqrt(2.)]
    ], dtype=torch.complex128)


def H(device='cpu'):
    return _H.to(device)


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
