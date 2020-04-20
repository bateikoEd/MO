import numpy as np

A_3_2 = np.array([
                 [1,0, 0,0,-0.5,-0.5,0],
                [0,1,-0.25,0,-0.25,0,-0.25],
                [0,-0.5,1,-0.5,0,0,0],
                [0,0,-0.5,1,-0.5,0,0],
                [0,-0.25,0,-0.25,1,0,-0.25],
                [0,0,0,0,0,1,-0.5],
                [0,-0.25,0,0,-0.25,-0.25,1]])
b_3_2 = np.array([0,0,0,0,0.25,0.5,0])

A_3_4 = np.array([
                 [1,-0.5,0,0,0,0,0,-0.5],
                [0,1,-0.25,0,-0.25,0,0,-0.25],
                [0,-0.5,1,-0.5,0,0,0,0],
                [0,0,-0.5,1,-0.5,0,0,0],
                [0,-0.25,0,-0.25,1,-0.25,0,-0.25],
                [0,0,0,0,-0.5,1,-0.5,0],
                [0,0,0,0,0,-0.5,1,-0.5],
                [0,-0.25,0,0,-0.25,0,-0.25,1]])
b_3_4 = np.array([1,1,1,1,1,1,1,1])

A_1 = np.array([[-2,1,1],[1,-2,1],[1,1,-2]])
b_1 = np.array([0,0,0])

A = A_1
b = b_1
print(f"A:\n{A}\nb:\n{b}")

result = np.linalg.tensorsolve(a=A, b=b)
print(f"result:\n{result}\nsum:\t{np.sum(result)}")