# 1. Variable and AutoGrad

## 1.3 Autograd Calculation

[Back to Home](https://github.com/taeoh-kim/Pytorch_Tutorial)

---

Default Mode is GPU Mode. If not, remove all GPU functions (ex. .cuda() or .cpu())

In this tutorial,

You will build Simple Feed-forward Computational Graph

and then, back-propagate scalar value

and Print parameters' gradients!

Finally, you will show what if you do not reset gradients of parameters.