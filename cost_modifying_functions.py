import torch


def identity(value):
    return value


def power_func(power):
    def func(x):
        return torch.float_power(torch.abs(x), power)
    return func


def root_func(power):
    def func(x):
        return x-x if x < 1e-2 else torch.float_power(torch.abs(x), 1/power)
    return func


def funky_func():
    def func(x):
        return 1-(torch.exp(-x*10))
    return func
