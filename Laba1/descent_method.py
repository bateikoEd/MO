import numpy as np
from numpy import linalg as npl

import functions as f

alpha_k = lambda x: np.float(
    np.dot(f.vector_h(x), f.vector_h(x)) / np.dot(np.dot(f.matrix_A, f.vector_h(x)), f.vector_h(x)))

count = 1
xk = np.array([0, 0], dtype=float)
vector_h1 = f.vector_h(xk)
alpha_k1 = alpha_k(xk)

const_func_xk_1 = np.float(f.func(xk + np.array(f.multiply_on_const(vector_h1, alph=alpha_k1))))

file_name = 'descent_method.txt'

with open(file_name, "w+") as file:
    line = f"\ncount = \t{count}\n xk =\t{xk}\n h =\t{vector_h1}\n alpha =\t {alpha_k1}"
    file.write(line)

    while npl.norm(vector_h1) > f.h:
        xk = xk + np.array(f.multiply_on_const(vector_h1, alph=alpha_k1))
        vector_h1 = f.vector_h(xk)
        alpha_k1 = alpha_k(xk)

        const_func_xk_1 = np.float(f.func(xk + np.array(f.multiply_on_const(vector_h1, alph=alpha_k1))))
        count += 1

        line = f"\ncount = \t{count}\n xk =\t{xk}\nfunc(xk) = {f.func(xk)}\n h =\t{vector_h1}\nalpha =\t {alpha_k1}"
        file.write(line)


print(f" count = \t{count}\nx* =\t{xk} \n func(xk) = {f.func(xk)}\n gradient(xk) = \t{f.gradient(xk)}")
