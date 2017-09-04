# MVP LAB Pytorch Tutorial
# Made by Taeoh Kim and Sungju Park
#
# 1. Variable and Autograd
# 1.1 Tensor and Variable

import torch
from torch.autograd import Variable

# ---------------------- GPU Process for Tensor & Models
if torch.cuda.is_available():
    print("\nCuda is Available \n")

# ---------------------- Define Zero Tensor
x = torch.Tensor(3,4)

# ---------------------- Tensor to Cuda Tensor
xc = x.cuda()

# ---------------------- And to Variable
xcv = Variable(xc)

print("x is : ")
print(x)

print("x cuda is : ")
print(xc)

print("x cuda Variable is : ")
print(xcv)

print("x cuda Variable Data is : ")
print(xcv.data)

print("x cuda Varialbe Grad is : ") # Gradient
print(xcv.grad)

print("\nx cuda Variable requires_grad is : ") # Requires Gradient
print(xcv.requires_grad)

# ---------------------- Data Access
print("\n\n")

xv = Variable(torch.rand(3,4))

print(xv)
print(xv.data)
print(xv[0]) # First Column
print(xv.data[0]) # Data's First Column

print("First Element: %f" % xv.data[0][0]) # First Column, First Row (Element)
print("Mean: %f" % xv.data.mean()) # Mean of All data
print("Sum: %f" % xv.data.sum()) # Sum of All data