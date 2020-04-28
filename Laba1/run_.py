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

# a, b, c, d, e, r, m = 2, 1, 3, 4, 1, 1, 1
# vector_const1 = [a, b, c]
# vector_const2 = [d, e, r, m]
# #
func = lambda x: np.float(2 * x[0] ** 2 + 8 * x[1] ** 2 - 0.01*x[0]*x[1] + x[0] - x[1])
# # substitution = lambda x : np.dot(x,vector_const2[:3]) - m
# #
# h = 10 ** (-3)
# alpha = 0.5
# #
# object = Optimization(func, h)
# # xk = np.array([1, -1, -2])
# xk = object.conjugated_gradient_method()
# print(f"xk:\t{xk}")

A = [[1,1,1]]
c = [1,2]
b = [1]
# x0 = [0,0,1,2]

object = Optimization(func,h=1e-5)
res = object.simplex_method(A,c,b)
