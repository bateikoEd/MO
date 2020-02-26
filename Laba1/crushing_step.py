import numpy as np
import pygal
from numpy import linalg as npl

import functions as f

"""const lambda = 1/2 , betta = 1"""
const_lambda = np.float(0.5)
const_betta = np.float(1)

xk = f.xk
'''xk started point in method'''
count = 1
vector_h1 = f.vector_h(xk)
alpha_k1 = const_betta
const_func_xk_1 = f.func(xk + f.multiply_on_const(vector_h1, alpha_k1))

file_name = "crushing_step.txt"

vector_x = [xk]

with open(file_name, "w+") as file:
    line = f"\ncount = \t{count}\n xk =\t{xk}\n h =\t{vector_h1}\n alpha =\t {alpha_k1}"
    file.write(line)

    while np.abs(const_func_xk_1 - f.func(xk)) > f.h:

        xk = xk + np.array(f.multiply_on_const(vector_h1, alph=alpha_k1))
        const_func_xk_1 = f.func(xk + f.multiply_on_const(vector_h1, alpha_k1))

        vector_h1 = f.vector_h(xk)
        alpha_k1 = np.dot(alpha_k1, const_lambda**2)
        count += 1

        vector_x.append(xk)

        line = f"\ncount = \t{count}\n xk =\t{xk}\nfunc(xk) = {f.func(xk)}\n h =\t{vector_h1} \nalpha =\t {alpha_k1}"
        file.write(line)

print(f"****count = \t{count}\nx* =\t{xk} \n func(xk) = {f.func(xk)}\n gradient(xk) = \t{f.gradient(xk)}")

xy_chart = pygal.XY()
xy_chart.title = 'crushing_step'
result = [(elem[0], elem[1]) for elem in vector_x]
xy_chart.add('dots', result)

xy_chart.render_to_file('crushing_step.svg')