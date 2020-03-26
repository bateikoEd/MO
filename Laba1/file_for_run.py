import numpy as np

from class_opimaz import Optimization
from class_opimaz import show_plot

func = lambda x: np.float(2 * x[0] ** 2 + 8 * x[1] ** 2 + 0.01 * x[0] * x[1] + x[0] - x[1])
h = 10 ** (-5)
alpha = 10 ** (-1)
optimazation_function_object = Optimization(func, h)
xk = [0, 0]
# object.with_constant_method(xk, alpha)

# print(object.double_derivation_function(xk))
x_res_grad = optimazation_function_object.with_constant_method(xk, alpha)
x_res_newton = optimazation_function_object.newton_method(xk)

print(f"gradient_method:\t{x_res_grad}\nnewthon_method:\t{x_res_newton}")
show_plot()