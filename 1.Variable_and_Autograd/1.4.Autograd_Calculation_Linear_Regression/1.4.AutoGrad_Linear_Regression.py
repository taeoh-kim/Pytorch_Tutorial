# MVP LAB Pytorch Tutorial
# Made by Taeoh Kim and Sungju Park
#
# 1. Variable and Autograd
# 1.4 Autograd Calculation in Linear Regression

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

num_epoch = 400
learning_rate = 0.01

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# ---------------------- Get Tensor
def get_data(x):
	if x.data.dim() >= 2:
		return x.data.mean()
	else:
		return x.data[0]

# ---------------------- Get Named Parameter Tensor
def get_param_data(x):
	if x[1].data.dim() >= 2:
		return x[1].data.mean()
	else:
		return x[1].data[0]

# ---------------------- Get Named Parameter Gradient
def get_grad(x):
	if x[1].grad is not None:
		return x[1].grad.data.mean()
	else:
		return "None"

# ---------------------- Get Named Parameter Name
def get_name(x):
	return x[0]

# ---------------------- Print Current Parameter and Gradient Value
def print_current_parameter_state():
	print("W Data: %f" % (get_param_data(param[0])))
	print("Gradient : %s" % (get_grad(param[0])))

# ---------------------- Training Input & Output
# ---------------------- Ground Truth : w = 2, b = 1
x = 5 * torch.randn(100, 1)
y = 2 * x + 1

x = to_variable(x)
y = to_variable(y)

# ---------------------- Define Model 1->1 Linear Network
model = nn.Linear(1, 1)
model = model.cuda()

print("\nModel is : ")
print(model)

# ---------------------- Obtain Parameters of Neural Network
param = list(model.named_parameters())

print("\nParameter List : ")
for i in param:
	print(i)

# ---------------------- Define Optimizer
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# ---------------------- Training Loop
for i in range(num_epoch):
	
	# ---------------------- Zero Grad to Model
	if i < 2:
		print("\n*** Before Zero_Grad ***")
		print_current_parameter_state()
	
	model.zero_grad()

	if i < 2:
		print("\n*** After Zero_Grad ***")
		print_current_parameter_state()

	# ---------------------- Forward, Loss, Backward
	y_pred = model(x)
	loss = torch.mean((y_pred - y) * (y_pred - y))
	loss.backward()

	if i < 2:
		print("\n*** After Loss Backward ***")
		print_current_parameter_state()

	# ---------------------- Optimizer Run
	optimizer.step()

	if i < 2:
		print("\n*** After Optimizer Step ***")
		print_current_parameter_state()

	# ---------------------- Print Los
	if i % 20 == 0:
		print("\nIterdation: [%d/%d], Loss: %f, '%s': %f, '%s': %f" % (i+1, num_epoch, get_data(loss), get_name(param[0]), get_param_data(param[0]), get_name(param[1]), get_param_data(param[1])))

x_test = to_variable(3 * torch.ones(1, 1))
y_test = model(x_test)

print("\n\nTest: 2 x 3 + 1 = %f" % (get_data(y_test)))