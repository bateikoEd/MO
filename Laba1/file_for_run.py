import numpy as np

from class_opimaz import Optimization

func = lambda x: np.float(2 * x[0] ** 2 + 8 * x[1] ** 2 + 0.01 * x[0] * x[1] + x[0] - x[1])
h = 10 ** (-8)
alpha = 10 ** (-2)
object = Optimization(func, h)
xk = [0, 0]
# object.with_constant_method(xk, alpha)

# print(object.double_derivation_function(xk))
object.newton_method(xk)