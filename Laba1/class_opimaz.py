import numpy as np
from numpy import linalg as npl
import matplotlib.pyplot as plt


def create_file(file_name='ok'):
    with open(file_name, "w+") as f:
        f.write(file_name)


def create_pot(x_label_name='x_label', y_label_name='y_label', plot_title='plot_title'):
    plt.xlabel(x_label_name)
    plt.ylabel(y_label_name)
    plt.title(plot_title)


def create_arrows(vector_x, name_text='none', arrow_style="->", color='r'):
    len_of_vector_x = len(vector_x)

    for i in range(0, len_of_vector_x - 1):
        plt.annotate(f"{i}", xytext=(vector_x[i][0], vector_x[i][1]), xy=(vector_x[i + 1][0], vector_x[i + 1][1]),
                     arrowprops=dict(arrowstyle=arrow_style, color=color))

    plt.xlim(vector_x[0][0] - 1, vector_x[len_of_vector_x - 1][0] + 1)
    plt.ylim(vector_x[1][0] - 1, vector_x[len_of_vector_x - 1][1] + 1)


def show_plot():
    plt.show()

def derivative_n(func, x, n, h=10**(-5)):
    if n == 0:
        return np.float(func(x))
    func1 = lambda y: np.float(0.5 * (func(y + h) - func(y - h)) / h)
    return derivative_n(func1, x, n - 1)


class Optimization():
    matrix_A = np.array([[4, -0.01], [-0.01, 16]], dtype=float)
    vector_b = np.array([1, -1], dtype=float)
    func = staticmethod(
        lambda x: np.float(0.5 * np.dot(np.dot(Optimization.matrix_A, x), x) + np.dot(Optimization.vector_b, x)))

    vector_h_for_gradient_method = staticmethod(lambda x: - Optimization.gradient(x))
    alpha_k_for_descent_method = staticmethod(lambda x: np.float(
        npl.norm(Optimization.vector_h_for_gradient_method(x)) ** 2 /
        np.dot(np.dot(Optimization.matrix_A, Optimization.vector_h_for_gradient_method(x)),
               Optimization.vector_h_for_gradient_method(x))))

    alpha_derivative = staticmethod(lambda x: x * 0.5)

    def __init__(self, func1,
                 h=np.float(10 ** (-3)),
                 matrix_A1=np.array([[4, -0.01], [-0.01, 16]], dtype=float),
                 vector_b=np.array([1, -1], dtype=float)):
        self.h = np.float(h)
        self.func = func1
        self.matrix_A = matrix_A1.copy()
        self.vector_b = vector_b.copy()

    def gradient(self, x):

        return np.array([(np.float(self.func([x[0] + self.h, x[1]]) -
                                   self.func([x[0] - self.h, x[1]])) / (
                                  2 * self.h)),
                         (np.float(self.func([x[0], x[1] + self.h]) -
                                   self.func([x[0], x[1] - self.h])) / (
                                  2 * self.h))])

    def gradient_3(self, x):
        return np.array([np.float((self.func([x[0] + self.h, x[1], x[2]]) -
                                   self.func([x[0] - self.h, x[1], x[2]])) / (
                                          2 * self.h)),
                         np.float((self.func([x[0], x[1] + self.h, x[2]]) -
                                   self.func([x[0], x[1] - self.h, x[2]])) / (
                                          2 * self.h)),
                         (np.float((self.func([x[0], x[1], x[2] + self.h]) -
                                    self.func([x[0], x[1], x[2] - self.h])) / (
                                           2 * self.h)))])

    def double_derivation_function(self, xk_vector):
        a11 = np.float((self.func([xk_vector[0] + self.h, xk_vector[1]]) -
                        2.0 * self.func(xk_vector) +
                        self.func([xk_vector[0] - self.h, xk_vector[1]])) / self.h ** 2)
        a22 = np.float((self.func([xk_vector[0], xk_vector[1] + self.h]) -
                        2.0 * self.func(xk_vector) +
                        self.func([xk_vector[0], xk_vector[1] - self.h])) / self.h ** 2)
        a21 = np.float((self.func([xk_vector[0] + self.h, xk_vector[1] + self.h]) -
                        self.func([xk_vector[0] - self.h, xk_vector[1] + self.h]) -
                        self.func([xk_vector[0] + self.h, xk_vector[1] - self.h]) +
                        self.func([xk_vector[0] - self.h, xk_vector[1] - self.h])) / (4 * self.h ** 2))

        return np.array([[a11, a21], [a21, a22]])

    def print_result(self, count, xk1, alpha):
        print(f" count = \t{count}\nx* =\t{xk1} \n func(xk) = {self.func(xk1)}\nalpha = \t{alpha}")

    def string_add_in_file(self, count, xk1, vector_h, alpha, file_name):
        line = f"\ncount = \t{count}\n xk =\t{xk1}\nfunc(xk) = {self.func(xk1)}\n h =\t{vector_h}\nalpha =\t {alpha}\n"
        with open(file_name, "a") as f:
            f.writelines(line)

    def string_add_in_file_projection(self, count, xk1, alpha, file_name):
        line = f"\ncount = \t{count}\n xk =\t{xk1}\nfunc(xk) = {self.func(xk1)}\nalpha =\t {alpha}\n"

        with open(file_name, "a") as f:
            f.writelines(line)

    def descent_method(self, xk1=None, file_name='descent_method.txt', color='r'):
        if xk1 is None:
            xk1 = [0, 0]

        xk_vector = xk1
        count = 1
        vector_h1 = self.vector_h_for_gradient_method(xk_vector)
        alpha_k1 = self.alpha_k_for_descent_method(xk_vector)

        const_func_xk_1 = np.float(self.func(xk_vector + np.array(vector_h1 * alpha_k1)))

        vector_x = [xk_vector]
        xk_1 = xk_vector

        create_file(file_name)
        self.string_add_in_file(count, xk_vector, vector_h1, alpha_k1, file_name)

        while np.abs(const_func_xk_1 - self.func(xk_vector)) > self.h:
            xk_vector = xk_vector + np.array(vector_h1 * alpha_k1)
            vector_h1 = self.vector_h_for_gradient_method(xk_vector)
            alpha_k1 = self.alpha_k_for_descent_method(xk_vector)

            const_func_xk_1 = np.float(self.func(xk_vector + np.array(vector_h1 * alpha_k1)))

            count += 1
            vector_x.append(xk_vector)
            print(self.print_result(count, xk_vector, alpha_k1))

            self.string_add_in_file(count, xk_vector, vector_h1, alpha_k1, file_name)

        # create arrows
        create_arrows(vector_x, color=color)

        return xk_vector

    def with_constant_method(self, xk1=None, alpha=np.float(10 ** (-3)), file_name='with_const_alpha.txt', color='r'):
        if xk1 is None:
            xk1 = [0, 0]

        xk_vector = xk1

        count = 1
        vector_h1 = self.vector_h_for_gradient_method(xk_vector)

        vector_x = [xk_vector]

        create_file(file_name)
        self.string_add_in_file(count, xk_vector, vector_h1, alpha, file_name)
        while npl.norm(vector_h1) > self.h:
            xk_vector = xk_vector + np.array(vector_h1 * alpha)
            vector_h1 = self.vector_h_for_gradient_method(xk_vector)
            count += 1

            vector_x.append(xk_vector)

            self.string_add_in_file(count, xk_vector, vector_h1, alpha, file_name)

        # create arrows
        create_arrows(vector_x, color=color)

        return xk_vector

    def crushing_step_method(self, xk1=None, const_lambda=np.float(0.5), const_betta=np.float(1),
                             file_name="crushing_step.txt", color='b'):
        if xk1 is None:
            xk1 = [0, 0]

        xk_vector = xk1
        '''xk started point in method'''
        count = 1
        vector_h1 = self.vector_h_for_gradient_method(xk_vector)
        alpha_k1 = np.float(const_betta * const_lambda)
        const_func_xk_1 = self.func(xk_vector + vector_h1 * alpha_k1)

        vector_x = [xk_vector]
        count_power = 1

        create_file(file_name)
        self.string_add_in_file(count, xk_vector, vector_h1, alpha_k1, file_name)

        while npl.norm(vector_h1) > self.h:

            xk_vector = xk_vector + np.array(vector_h1 * alpha_k1)
            const_func_xk_1 = self.func(xk_vector + vector_h1 * alpha_k1)

            vector_h1 = self.vector_h_for_gradient_method(xk_vector)
            count += 1
            if np.abs(const_func_xk_1 - self.func(xk_vector)) > self.h:
                alpha_k1 = np.dot(const_betta, const_lambda ** 2)
                count_power += 1

            vector_x.append(xk_vector)

            self.string_add_in_file(count, xk_vector, vector_h1, alpha_k1, file_name)

        # create arrows
        create_arrows(vector_x, color=color)

        return xk_vector

    def newton_method(self, xk1=None, alpha=1, epsilon=0.01, file_name='newton_method.txt', color='g'):
        if xk1 is None:
            xk1 = np.zeros(2)

        double_derivation_matrix = self.double_derivation_function(xk1)
        vector_h1 = npl.tensorsolve(double_derivation_matrix, - self.gradient(xk1))

        xk_vector = xk1 + vector_h1
        count = 1

        vector_x = [xk1, xk_vector]
        const_func_xk_1 = self.func(xk_vector)

        condition_const = np.abs(const_func_xk_1 - self.func(xk1) -
                                 epsilon * alpha * np.dot(self.gradient(xk1), vector_h1))

        create_file(file_name)
        self.string_add_in_file(count, xk_vector, vector_h1, alpha, file_name)

        while condition_const > self.h:
            alpha *= 0.5

            double_derivation_matrix = self.double_derivation_function(xk_vector)
            vector_h1 = npl.tensorsolve(double_derivation_matrix, - self.gradient(xk_vector))

            xk_vector = xk_vector + vector_h1 * alpha

            const_func_xk_1 = self.func(xk_vector + vector_h1 * alpha)

            condition_const = np.abs(const_func_xk_1 - self.func(xk_vector) -
                                     epsilon * alpha * np.dot(self.gradient(xk_vector), vector_h1))
            count += 1
            vector_x.append(xk_vector)

            print(self.print_result(count, xk_vector, alpha))

            self.string_add_in_file(count, xk_vector, vector_h1, alpha, file_name)

        # create arrows
        create_arrows(vector_x, color=color)

        return xk_vector

    def gradient_projection_method(self, xk=None, vector_const2=None, alpha=1, file_name="projection_mathod.txt"):
        if xk is None:
            xk = np.zeros(3)
        if vector_const2 is None:
            vector_const1 = np.array([1., 1, 1, 1])

        create_file(file_name=file_name)

        xk = np.array(xk)
        vector_const2 = np.array(vector_const2)

        vector_for_xk = [xk]

        count = 0
        xk_1 = xk
        self.string_add_in_file_projection(count=count, xk1=xk, file_name=file_name,alpha=alpha)

        count = 1
        # alpha = self.alpha_derivative(alpha)
        const_1 = 1./(vector_const2[0] ** 2 + vector_const2[1] ** 2 + vector_const2[2] ** 2)
        my_lambda = const_1 * (vector_const2[3] - np.dot(vector_const2[:3], xk))

        grad = self.gradient_3(xk)

        xk_multi_on_const = xk - grad * alpha
        lambda_mult_const2 = np.array(vector_const2[0:3]) * my_lambda

        xk = xk_multi_on_const + lambda_mult_const2

        vector_for_xk.append(xk)
        self.string_add_in_file_projection(count=count, xk1=xk, file_name=file_name,alpha=alpha)

        # print(f"lambda_k:\t{my_lambda}\talpha_k:\t{alpha}\ngrad_k:{grad}\n"
        #       f"xk-alpha*grad_k:\t{xk_multi_on_const}\n"
        #       f"const_2_on_xk:\t{lambda_mult_const2}\n"
        #       f"res:\t{xk}\nfunc:\t{self.func(xk)}\n")

        while npl.norm(xk - xk_1) > self.h:
            xk_1 = xk

            alpha = self.alpha_derivative(alpha)

            const_1 = np.float(1./(vector_const2[0] ** 2 + vector_const2[1] ** 2 + vector_const2[2] ** 2))
            my_lambda = np.float(const_1 * (vector_const2[3] - np.dot(vector_const2[:3], xk)))

            grad = self.gradient_3(xk)

            xk_multi_on_const = xk - grad * alpha
            lambda_mult_const2 = np.array(vector_const2[0:3]) * my_lambda

            xk = xk_multi_on_const + lambda_mult_const2

            count += 1
            self.string_add_in_file_projection(count=count, xk1=xk, file_name=file_name, alpha=alpha)

            vector_for_xk.append(xk)

        create_arrows(vector_for_xk)

        return xk

    def gradient_method_for_one_dimention(self, func, xk=None, alpha=1):
        if xk is None:
            xk = 5
        hk = - derivative_n(func=func, x=xk, n=1)
        xk = xk + 0.5 * alpha * hk

        print(f"xk:\t{xk}\tfunc:{func(xk)}")

        while np.abs(hk) > self.h :
            hk = - derivative_n(func=func, x=xk, n=1)
            xk = xk + 0.5 * alpha * hk

            print(f"xk:\t{xk}\tfunc:{func(xk)}")

        if xk > 0:
            return xk
        return 10**(-3)

    def conjugated_gradient_method(self, xk0=None, file_name='conjugated_gradient_method.txt', color='r'):
        if xk0 is None:
            xk0 = np.zeros(2)

        n = len(xk0)

        hk = - self.gradient(xk0)

        # func1 = lambda x: self.func(xk0 + x * hk)

        alpha_k = self.gradient_method_for_one_dimention(func=lambda x: self.func(xk0 + x * hk))

        xk_1 = xk0
        xk = xk0 + alpha_k * hk

        print(f"count:\t{0}\txk:\t{xk}\talpha:\t{alpha_k}\tfunc:\t{self.func(xk)}\t")

        count = 1
        while npl.norm(hk) > self.h:

            grad_xk = self.gradient(xk)
            if count % n != 0 :
                betta_k = (1/npl.norm(grad_xk)**2) * np.dot(grad_xk, grad_xk - self.gradient(xk_1))
            else:
                betta_k = 0

            func1 = lambda ak: self.func(xk + ak * hk)

            hk = - grad_xk + betta_k * hk

            alpha_k = self.gradient_method_for_one_dimention(func1)

            xk_1 = xk
            xk = xk + alpha_k * hk

            print(f"count:\t{count}\txk:\t{xk}\talpha:\t{alpha_k}\tbetta_k:\t{betta_k}\tfunc:\t{self.func(xk)}\t")

            count += 1


        return xk
    def symplex_method(self,A,c,b,x0):
        A = np.array(A)
        c = np.array(c)
        b = np.array(b)
        x0 = np.array(x0)

        m = A.shape[0]
        n = A.shape[1]
        l = c.size

        count_of_basis_vectors = np.array([x for x in range(m,l)])
        if l > m:
            basis = np.identity(l - m)
            A = np.append(A, basis, axis=1)

        print(f"m:\t{m}\nn:{n}\nl:\t{l}\nA:\n{A}\nb:\n{b}\ncount_of:\n{count_of_basis_vectors}")
        print(f"count[0]:\t{count_of_basis_vectors[0]}\tcount[1]:\t{count_of_basis_vectors[-1]}\nshape:\t{(n,1)}")
        delta = np.sum(A * np.reshape(c[count_of_basis_vectors[0] : count_of_basis_vectors[-1] + 1] , (n,1)), axis=0) - c

        max_index = np.where(delta == np.max(delta))[0][0]

        # find count of vector with need to swap
        x_divide_on_coeficients = np.array([elem[max_index] for elem in A])

        temp = b/ x_divide_on_coeficients
        min_in_swap = np.min(temp)
        swap_index = np.where(temp == min_in_swap)[0][0]

        basic_vector = np.array([elem[swap_index] for elem in A])

        count_of_basis_vectors = np.delete(count_of_basis_vectors, swap_index)
        count_of_basis_vectors = np.append(count_of_basis_vectors, max_index)
        count_of_basis_vectors = np.sort(count_of_basis_vectors)

        for i, elem in enumerate(A):
            if i == swap_index:
                A[i] = A[i] / A[swap_index][max_index]
                b[i] = b[i] / A[swap_index][max_index]
                continue
            A[i] = A[i] - A[i][max_index] * A[swap_index]
            b[i] = b[i] - A[i][max_index] * b[swap_index]

        delta = np.sum(A * np.reshape(c[count_of_basis_vectors[0] : count_of_basis_vectors[-1] + 1], (n,1)), axis=0) - c
        print(f"A\n{A}\ndelta:\n{delta}")



