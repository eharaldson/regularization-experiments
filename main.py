import matplotlib.pyplot as plt
import numpy as np

from sklearn import model_selection, linear_model

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
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    return x_train, x_test, y_train, y_test

x = np.random.rand(20,1)
x_train, x_test, y_train, y_test = create_polynomial_labels(x)

x_train_1 = x_train[:,0]
x_train_1 = x_train_1.reshape(-1,1)

print(x_train_1.shape)

model = linear_model.LinearRegression()
model.fit(x_train_1, y_train)

y_prediction = model.predict(x_train_1)

plt.scatter(x_train_1, y_train)
plt.plot(x_train_1, y_prediction, c='orange')
plt.show()