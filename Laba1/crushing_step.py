import numpy as np
from numpy import linalg as npl

import functions as f


const_lambda = np.float(0.5)
const_betta = np.float(1)

count = 1
xk = np.array([0, 0], dtype=float)
vector_h1 = f.vector_h(xk)
alpha_k1 = const_betta
# alpha_k1 = np.float(0.5)
const_func_xk_1 = f.func(xk + f.multiply_on_const(vector_h1, alpha_k1))

print(f"\ncount = \t{count}\n xk =\t{xk}\n h =\t{vector_h1}\n "
      f"alpha =\t {alpha_k1}\n const_func = \t{const_func_xk_1}")

while np.abs(const_func_xk_1 - f.func(xk)) > f.h:

    xk = xk + np.array(f.multiply_on_const(vector_h1, alph=alpha_k1))
    const_func_xk_1 = f.func(xk + f.multiply_on_const(vector_h1, alpha_k1))

    vector_h1 = f.vector_h(xk)
    alpha_k1 = np.dot(alpha_k1, const_lambda)

    count += 1
    if count < 5:
        print(f"\ncount = \t{count}\n xk =\t{xk}\nfunc(xk) = {f.func(xk)}\n "
              f"h =\t{vector_h1}\n alpha =\t {alpha_k1}\n const_func = \t{const_func_xk_1} ")

print(f"****count = \t{count}\nx* =\t{xk} \n func(xk) = {f.func(xk)}\n gradient(xk) = \t{f.gradient(xk)}")
print(f"func1:\t{f.func(xk)}\nfunc2:\t{f.func([2.1, 1])}")
