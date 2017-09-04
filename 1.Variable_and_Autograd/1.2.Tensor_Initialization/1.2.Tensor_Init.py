# MVP LAB Pytorch Tutorial
# Made by Taeoh Kim and Sungju Park
#
# 1. Variable and Autograd
# 1.2 Tensor Initialization

import torch
import numpy as np
from torch.autograd import Variable

# ---------------------- Function for Cuda & Variable Wrapping
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# Various Torch Initialized Methods
x_eye = to_variable(torch.eye(3,3))
x_linspace = to_variable(torch.linspace(0,2,10))
x_ones = to_variable(torch.ones(2,3))
x_zeros = to_variable(torch.zeros(2,5))
x_uniform = to_variable(torch.rand(2,4))
x_gaussian = to_variable(torch.randn(4,2))

print("Eye : ", x_eye)
print("Linspace : ", x_linspace)
print("Ones : ", x_ones)
print("Zeros : ", x_zeros)
print("Rand_Uniform : ", x_uniform)
print("Rand_Gaussian : ", x_gaussian)
print("\n")

# And From Numpy
a = np.array([[1, 2, 3], [4, 5, 6]])
x_numpy = to_variable(torch.from_numpy(a))

print("From Numpy : ", x_numpy)

# And to Tensor Again
print("As Numpy : ", x_numpy.data.cpu().numpy())