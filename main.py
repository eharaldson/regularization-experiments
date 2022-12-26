import matplotlib.pyplot as plt
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

def create_polynomial_labels(x):

    x = expand_power_of_values(x, 3)
    weights = np.array([1, 3, -6])
    bias = 2
    y = x @ weights + bias
    return y

x = np.random.rand(20,1)
y = create_polynomial_labels(x)

plt.scatter(x, y)
plt.show()