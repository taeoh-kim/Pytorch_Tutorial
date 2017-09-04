# MVP LAB Pytorch Tutorial
# Made by Taeoh Kim and Sungju Park
#
# 1. Variable and Autograd
# 1.3 Autograd Calculation

import torch
import numpy as np
from torch.autograd import Variable

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad = True)

# ---------------------- Get Grad
def get_grad(x):
	if x.grad is not None:
		return x.grad.data.mean()
	else:
		return "None"

# ---------------------- 4 Input Nodes
a = np.array([1, 2, 3])
b = np.array([-1, -2, -3])
c = np.array([-10, 0, -5])

x1 = to_variable(torch.from_numpy(a))
x3 = to_variable(torch.from_numpy(b))
x4 = to_variable(torch.from_numpy(c))

# ---------------------- Computation Graph (Forward)
x2 = x3 + 2 * x4
# dx2/dx3 = 1
# dx2/dx4 = 2

y = 2 * x1 + 3 * x2 + 1
# dy/dx1 = 2
# dy/dx2 = 3

print(y.data)

# ---------------------- Unit Backward
y.backward(torch.Tensor([1.0]))
print("Unit Backward")

# ---------------------- Auto-Calculated Gradient
print("x1 grad = %s" % (get_grad(x1))) # dy/dx1 = 2
print("x2 grad = %s" % (get_grad(x2))) # It is Not Input NODE
print("x3 grad = %s" % (get_grad(x3))) # dy/dx3 = dy/dx2 * dx2/dx3 = 3 * 1 = 3
print("x4 grad = %s" % (get_grad(x4))) # dy/dx4 = dy/dx2 * dx2/dx4 = 3 * 2 = 6

# ---------------------- Double Backward
y.backward(torch.Tensor([2.0]))
print("\nDouble Backward")

# ---------------------- Auto-Calculated Gradient
print("x1 grad = %s" % (get_grad(x1))) # 4?
print("x2 grad = %s" % (get_grad(x2))) # It is Not Input NODE
print("x3 grad = %s" % (get_grad(x3))) # 6?
print("x4 grad = %s" % (get_grad(x4))) # 12?

# ---------------------- Forward Again
x1.grad.data.zero_()
x3.grad.data.zero_()
x4.grad.data.zero_()

# ---------------------- Double Backward
y.backward(torch.Tensor([2.0]))
print("\nDouble Backward After Reset")

# ---------------------- Auto-Calculated Gradient
print("x1 grad = %s" % (get_grad(x1))) # 4
print("x2 grad = %s" % (get_grad(x2))) # It is Not Input NODE
print("x3 grad = %s" % (get_grad(x3))) # 6
print("x4 grad = %s" % (get_grad(x4))) # 12