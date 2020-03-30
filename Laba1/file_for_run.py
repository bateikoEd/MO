import numpy as np

from class_opimaz import Optimization
from class_opimaz import show_plot

'''
gradient projection method: 
 F(x,y,z) = a*x^2 + b*y^2 + c*z^2
X = {(x,y,z): d*x + e*y + r*z = m }

vector_const1 = (a,b,c)
vector_const2 = (d,e,r,m)
'''

a, b, c, d, e, r, m = 2, 1, 3, 4, 1, 1, 1
vector_const1 = [a, b, c]
vector_const2 = [d, e, r, m]

func = lambda x: np.float(a * x[0] ** 2 + b * x[1] ** 2 + c * x[2] ** 2)
h = 10 ** (-4)
alpha = 1
optimazation_function_object = Optimization(func, h)
xk = np.array([0, 0, 0])

# print(f"gradient_method:\t{x_res_grad}\nnewthon_method:\t{x_res_newton}")
# show_plot()

xk = optimazation_function_object.gradient_projection_method(xk, vector_const1=vector_const1, vector_const2=vector_const2,
                                                             alpha=alpha)
print(xk)