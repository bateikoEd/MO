import numpy as np
from numpy import linalg as npl

def mult_on_const(x,alph=1.):
    y = []
    for elem in x:
        y.append(elem*alph)
    return y

h = np.float(10**(-10))

matrix_A = np.array([[2, 0], [0, 4]], dtype=float)
vector_b = np.array([-4,-4], dtype=float)

# func = lambda x: np.float(x[0]**2 + 18 * x[1]**2 + 0.01 * x[0] * x[1] + x[0] - x[1])

func = lambda x: np.float(np.dot(np.dot(matrix_A, x), x ) + np.dot(vector_b, x))

gradient = lambda x: np.array ( [ (func([x[0] + h, x[1]]) - func([x[0] - h, x[1]]))/(2.*h),
              (func([x[0], x[1] + h]) - func([x[0], x[1] - h]))/(2.*h) ], dtype=float)
vector_h = lambda x:  np.array(- np.dot(matrix_A, x) - vector_b)
# alpha_k = lambda x: np.float(- np.dot(gradient(x), vector_h(x)) / np.dot(np.dot(matrix_A, x), x))
alpha_k = lambda x: np.float(np.dot(vector_h(x), vector_h(x))/np.dot(np.dot(matrix_A, vector_h(x)), vector_h(x)))


count = 1
xk = np.array([0,0], dtype=float)
vector_h1 = vector_h(xk)
alpha_k1 = alpha_k(xk)

print(f"\ncount = \t{count}\n xk =\t{xk}\n h =\t{vector_h1}\n alpha =\t {alpha_k1}")

while count < 30:
    xk = xk + np.array( mult_on_const(vector_h1, alph=alpha_k1))
    vector_h1 = vector_h(xk)
    alpha_k1 = alpha_k(xk)
    count += 1
    print(f"\ncount = \t{count}\n xk =\t{xk}\nfunc(xk) = {func(xk)}\n h =\t{vector_h1}\n alpha =\t {alpha_k1}")

print(f" count = \t{count}\nx* =\t{xk} \n func(xk) = {func(xk)}\n gradient(xk) = \t{gradient(xk)}")

print(f"func :\t{func([0,0])}\n gradient : \t{gradient([0,0])}")
# print(f"func:\t{func([1,1])}\n func1:\t{func1([1,1])}")

