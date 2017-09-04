# 1. Variable and AutoGrad

## 1.1 Tensor and Variable

[Back to Home](https://github.com/taeoh-kim/Pytorch_Tutorial)

---

Default Mode is GPU Mode. If not, remove all GPU functions (ex. .cuda() or .cpu())

In this tutorial,

What is Tensor? Cuda Tensor? and Variable?

The most important one is "Variable".

It contains parameters such as data, grad and requires_grad.

data stores the data (scalar, matrix or tensor)

grad stores the gradient of current parameter (if current parameter is trainable)

And you will learn how to extract such data and gradient of the Variable.
