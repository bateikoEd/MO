import numpy as np
from numpy import linalg as npl
import pygal
import matplotlib.pyplot as plt

import functions as f

'''xk started point in method'''
"""alpha is a constant"""
xk = f.xk

count = 1
vector_h1 = f.vector_h(xk)
alpha_k1 = np.float(0.0001)

file_name = 'with_const_alpha.txt'

vector_x = [f.xk]

with open(file_name, "w+") as file:
    line = f"\ncount = \t{count}\n xk =\t{xk}\n h =\t{vector_h1}\n alpha =\t {alpha_k1}"
    file.write(line)

    while npl.norm(vector_h1) > f.h:
        xk = xk + np.array(f.multiply_on_const(vector_h1, alph=alpha_k1))
        vector_h1 = f.vector_h(xk)
        count += 1

        vector_x.append(xk)

        #print(f"xk:\t{xk}\ncount:\t{count}")

        line = f"\ncount = \t{count}\n xk =\t{xk}\nfunc(xk) = {f.func(xk)}\n h =\t{vector_h1}\nalpha =\t {alpha_k1}"
        file.write(line)


print(f" count = \t{count}\nx* =\t{xk} \n func(xk) = {f.func(xk)}\n gradient(xk) = \t{f.gradient(xk)}")

xy_chart = pygal.XY()
xy_chart.title = 'with_const_alpha'
result = [(elem[0], elem[1]) for elem in vector_x]
xy_chart.add('dots', result)

xy_chart.render_to_file('with_constant.svg')
