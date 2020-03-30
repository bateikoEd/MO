import numpy as np
h = np.float(10 ** (-3))

def derivative_n(func, x, n):
    if n == 0:
        return np.float(func(x))
    func1 = lambda y: np.float(0.5 * (func(y + h) - func(y - h)) / h)
    return derivative_n(func1, x, n - 1)

my_function = lambda x: np.exp(3*x)
print(f"{derivative_n(my_function, 0,3)}")

# def derivative_x_n(func, x, param_x, param_y):
#     if param_x == 0 and param_y == 0:
#         return np.float(func(x))
#
#     func1 = lambda y: np.float(0.5 * (func(y + h) - func(y - h)) / h)
#     return derivative_n(func1,  param_x - 1)
#
#
#
# print(f"{derivative_x_n(my_function, 3)(0)}")

# def derivative_of_parameter(func, x, param_x,param_y):



