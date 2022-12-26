import numpy as np

def expand_squared_values(x):
    return np.hstack((x, np.square(x)))

matrix = np.random.rand(20,1)

print(matrix)
print(expand_squared_values(matrix))