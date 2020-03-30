import numpy as np
from numpy import linalg as npl
import matplotlib.pyplot as plt


def multiply_on_const(x, alph=1.):
    y = []
    for elem in x:
        y.append(elem * alph)
    return np.array(y)


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


class Optimization():

    matrix_A = np.array([[4, -0.01], [-0.01, 16]], dtype=float)
    vector_b = np.array([1, -1], dtype=float)
    h = np.float(10)
    func = staticmethod(
        lambda x: np.float(0.5 * np.dot(np.dot(Optimization.matrix_A, x), x) + np.dot(Optimization.vector_b, x)))
    gradient = staticmethod(lambda x: np.array([(np.float(Optimization.func([x[0] + Optimization.h, x[1]]) -
                                                          Optimization.func([x[0] - Optimization.h, x[1]])) / (
                                                             2 * Optimization.h)),
                                                (np.float(Optimization.func([x[0], x[1] + Optimization.h]) -
                                                          Optimization.func([x[0], x[1] - Optimization.h])) / (
                                                             2 * Optimization.h))]))

    gradient_3 = staticmethod(lambda x: np.array([(np.float(Optimization.func([x[0] + Optimization.h, x[1], x[2]]) -
                                                          Optimization.func([x[0] - Optimization.h, x[1], x[2]])) / (
                                                             2 * Optimization.h)),
                                            (np.float(Optimization.func([x[0], x[1] + Optimization.h, x[2]]) -
                                                          Optimization.func([x[0], x[1] - Optimization.h], x[2])) / (
                                                             2 * Optimization.h)),
                                              (np.float(Optimization.func([x[0], x[1], x[2] + Optimization.h]) -
                                                       Optimization.func([x[0], x[1], x[2] - Optimization.h])) / (
                                                      2 * Optimization.h))]))

    vector_h_for_gradient_method = staticmethod(lambda x: - Optimization.gradient(x))
    alpha_k_for_descent_method = staticmethod(lambda x: np.float(
        npl.norm(Optimization.vector_h_for_gradient_method(x)) ** 2 /
        np.dot(np.dot(Optimization.matrix_A, Optimization.vector_h_for_gradient_method(x)),
               Optimization.vector_h_for_gradient_method(x))))

    alpha_derivative = staticmethod(lambda x: x*0.5)

    def __init__(self, func1,
                 h=np.float(10 ** (-3)),
                 matrix_A1=np.array([[4, -0.01], [-0.01, 16]], dtype=float),
                 vector_b=np.array([1, -1], dtype=float)):
        self.h = np.float(h)
        self.func = func1
        self.matrix_A = matrix_A1.copy()
        self.vector_b = vector_b.copy()

    def double_derivation_function(self, xk_vector):
        a11 = np.float((self.func([xk_vector[0] + self.h, xk_vector[1]]) -
                        2.0 * self.func(xk_vector) +
                       self.func([xk_vector[0] - self.h, xk_vector[1]])) / self.h**2)
        a22 = np.float((self.func([xk_vector[0], xk_vector[1] + self.h]) -
                        2.0 * self.func(xk_vector) +
                       self.func([xk_vector[0], xk_vector[1] - self.h])) / self.h**2)
        a21 = np.float((self.func([xk_vector[0] + self.h, xk_vector[1] + self.h]) -
                        self.func([xk_vector[0] - self.h, xk_vector[1] + self.h]) -
                        self.func([xk_vector[0] + self.h, xk_vector[1] - self.h]) +
                        self.func([xk_vector[0] - self.h, xk_vector[1] - self.h])) / (4 * self.h**2))

        return np.array([[a11, a21], [a21, a22]])

    def print_result(self, count, xk1, alpha):
        print(f" count = \t{count}\nx* =\t{xk1} \n func(xk) = {self.func(xk1)}\n gradient(xk) = "
              f"\t{self.gradient(xk1)}\n alpha = \t{alpha}")

    def string_add_in_file(self, count, xk1, vector_h, alpha, file_name):
        line = f"\ncount = \t{count}\n xk =\t{xk1}\nfunc(xk) = {self.func(xk1)}\n h =\t{vector_h}\nalpha =\t {alpha}"

        with open(file_name, "a") as f:
            f.writelines(line)

    def descent_method(self, xk1=None, file_name='descent_method.txt', color='r'):
        if xk1 is None:
            xk1 = [0, 0]

        xk_vector = xk1
        count = 1
        vector_h1 = self.vector_h_for_gradient_method(xk_vector)
        alpha_k1 = self.alpha_k_for_descent_method(xk_vector)

        const_func_xk_1 = np.float(self.func(xk_vector + np.array(multiply_on_const(vector_h1, alph=alpha_k1))))

        vector_x = [xk_vector]
        xk_1 = xk_vector

        create_file(file_name)
        self.string_add_in_file(count, xk_vector, vector_h1, alpha_k1, file_name)

        while np.abs(const_func_xk_1 - self.func(xk_vector)) > self.h:
            xk_vector = xk_vector + np.array(multiply_on_const(vector_h1, alph=alpha_k1))
            vector_h1 = self.vector_h_for_gradient_method(xk_vector)
            alpha_k1 = self.alpha_k_for_descent_method(xk_vector)

            const_func_xk_1 = np.float(self.func(xk_vector + np.array(multiply_on_const(vector_h1, alph=alpha_k1))))

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
            xk_vector = xk_vector + np.array(multiply_on_const(vector_h1, alph=alpha))
            vector_h1 = self.vector_h_for_gradient_method(xk_vector)
            count += 1

            vector_x.append(xk_vector)

            self.string_add_in_file(count, xk_vector, vector_h1, alpha, file_name)

        # create arrows
        create_arrows(vector_x,color=color)

        return xk_vector

    def crushing_step_method(self, xk1=None, const_lambda=np.float(0.5),  const_betta=np.float(1),
                             file_name="crushing_step.txt", color='b'):
        if xk1 is None:
            xk1 = [0, 0]

        xk_vector = xk1
        '''xk started point in method'''
        count = 1
        vector_h1 = self.vector_h_for_gradient_method(xk_vector)
        alpha_k1 = np.float(const_betta * const_lambda)
        const_func_xk_1 = self.func(xk_vector + multiply_on_const(vector_h1, alpha_k1))

        vector_x = [xk_vector]
        count_power = 1

        create_file(file_name)
        self.string_add_in_file(count, xk_vector, vector_h1, alpha_k1, file_name)

        while npl.norm(vector_h1) > self.h:

            xk_vector = xk_vector + np.array(multiply_on_const(vector_h1, alph=alpha_k1))
            const_func_xk_1 = self.func(xk_vector + multiply_on_const(vector_h1, alpha_k1))

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
            xk1 = [0, 0]

        double_derivation_matrix = self.double_derivation_function(xk1)
        vector_h1 = npl.tensorsolve(double_derivation_matrix, - self.gradient(xk1))

        xk_vector = xk1 + vector_h1
        count = 1

        vector_x = [xk1, xk_vector]
        const_func_xk_1 = self.func(xk_vector)

        condition_const = np.abs(const_func_xk_1 - self.func(xk1) -
                                 epsilon*alpha*np.dot(self.gradient(xk1), vector_h1))

        create_file(file_name)
        self.string_add_in_file(count, xk_vector, vector_h1, alpha, file_name)

        while condition_const > self.h:
            alpha *= 0.5

            double_derivation_matrix = self.double_derivation_function(xk_vector)
            vector_h1 = npl.tensorsolve(double_derivation_matrix, - self.gradient(xk_vector))

            xk_vector = xk_vector + multiply_on_const(vector_h1, alpha)

            const_func_xk_1 = self.func(xk_vector + multiply_on_const(vector_h1, alpha))

            condition_const = np.abs(const_func_xk_1 - self.func(xk_vector) -
                                     epsilon * alpha * np.dot(self.gradient(xk_vector), vector_h1))
            count += 1
            vector_x.append(xk_vector)

            print(self.print_result(count, xk_vector, alpha))

            self.string_add_in_file(count, xk_vector, vector_h1, alpha, file_name)

        # create arrows
        create_arrows(vector_x, color=color)

        return xk_vector


    def gradient_projection_method(self, xk=None, vector_const1=None, vector_const2=None, alpha=1):
        if xk is None:
            xk = np.array([0., 0, 0])
        if vector_const1 is None:
            vector_const1 = np.array([1., 1, 1])
        if vector_const2 is None:
            vector_const1 = np.array([1., 1, 1, 1])

        xk = np.array(xk)
        vector_for_xk = [xk]

        print(f"xk:\t{xk}\nvector1:\t{vector_const1}\nvector2:\t{vector_const2}\nalpha:\t{alpha}")

        while npl.norm(self.gradient(xk)) > self.h:
            alpha = self.alpha_derivative(alpha)
            const_1 = 1./(vector_const2[0] ** 2 + vector_const2[1] ** 2 + vector_const2[2] ** 2)
            my_lambda = const_1 * vector_const2[3] - const_1 * np.dot(vector_const2[:3], xk)

            print(f"alpha:\t{alpha}\tconst_1:\t{const_1}\tmy_lambda:\t{my_lambda}")

            xk = xk - multiply_on_const(self.gradient_3(xk), alph=alpha) + \
                 multiply_on_const(vector_const2[0:3], alph=my_lambda)

            print(f"xk:\t{xk}\tfunc:\t{self.func(xk)}")

            vector_for_xk.append(xk)

        return xk