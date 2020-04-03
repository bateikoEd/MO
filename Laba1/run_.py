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
substitution = lambda x : np.dot(x,vector_const2[:3]) - m

h = 10 ** (-10)
alpha = 0.5

object = Optimization(func, h)
xk = np.array([1, -1, -2])

# print(f"gradient_method:\t{x_res_grad}\nnewthon_method:\t{x_res_newton}")
# show_plot()
print(f"xo:\t{xk}")
xk = object.gradient_projection_method(xk, vector_const2=vector_const2,
                                                              alpha=alpha)
show_plot()
print(substitution(xk))