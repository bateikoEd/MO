import numpy as np
import numpy.matlib as npml

def multiply_on_const(x, alph=1.):
    y = []
    for elem in x:
        y.append(elem * alph)
    return y


h = np.float(10 ** (-10))
matrix_A = np.array([[4, -0.01], [-0.01, 16]], dtype=float)
vector_b = np.array([1, -1], dtype=float)
xk = np.array([0, 1], dtype=float)

func = lambda x: np.float(0.5 * np.dot(np.dot(matrix_A, x), x) + np.dot(vector_b, x))

gradient = lambda x: np.array([(func([x[0] + h, x[1]]) - func([x[0] - h, x[1]])) / (2 * h),
                               (func([x[0], x[1] + h]) - func([x[0], x[1] - h])) / (2 * h)],
                              dtype=float)
# vector_h = lambda x: np.array(- np.dot(matrix_A, x) - vector_b)
vector_h = lambda x: - gradient(x)