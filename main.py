import numpy as np

def expand_squared_values(x):
    return np.hstack((x, np.square(x)))

def expand_power_of_values(x, power):
    return_x = x
    for i in range(2,power+1):
        x_to_add = x ** i
        x_to_add = x_to_add.reshape(-1,1)
        return_x = np.concatenate((return_x, x_to_add), axis=1)
    return return_x

matrix = np.random.rand(20,1)

matrix = expand_power_of_values(matrix, 4)
print(matrix)