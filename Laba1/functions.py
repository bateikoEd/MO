import numpy as np
import numpy.matlib as npml

def multiply_on_const(x, alph=1.):
    y = []
    for elem in x:
        y.append(elem * alph)
    return y


h = np.float(10 ** (-3))
matrix_A = np.array([[4, -0.01], [-0.01, 16]], dtype=float)
vector_b = np.array([1, -1], dtype=float)
xk = np.array([0, 1], dtype=float)

# func = lambda x: np.float(0.5 * np.dot(np.dot(matrix_A, x), x) + np.dot(vector_b, x))
func = lambda x: np.float(2 * x[0]**2 + 8 * x[1]**2 + 0.01*x[0]*x[1] + x[0] - x[1])

# func = lambda x: np.float(3*(x[0] - 5)**2 + 4500*(x[1] + 6)**2 + 2*(x[0] - 5)*(x[1] + 6))
# func = lambda x: np.float(np.exp(x[0]**2 + x[1]**2))
gradient = lambda x: np.array([(np.float(func([x[0] + h, x[1]]) - func([x[0] - h, x[1]])) / (2 * h)),
                               (np.float(func([x[0], x[1] + h]) - func([x[0], x[1] - h])) / (2 * h))],
                              dtype=float)
# vector_h = lambda x: np.array(- np.dot(matrix_A, x) - vector_b)
vector_h = lambda x: - gradient(x)