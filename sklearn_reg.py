from sklearn import datasets, model_selection, linear_model, preprocessing
import numpy as np

np.random.RandomState(10)
# Load dataset
X, y = datasets.load_diabetes(return_X_y=True)

# Split data
X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.4)

# Normalize data
scaler = preprocessing.MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

# Create linear models
model = linear_model.LinearRegression().fit(X_train, y_train)
model_L1 = linear_model.Lasso(alpha=0.1).fit(X_train, y_train)
model_L2 = linear_model.Ridge(alpha=0.1).fit(X_train, y_train)

print(f'No regularisation: {model.score(X_val, y_val)}')
print(f'L1 regularisation: {model_L1.score(X_val, y_val)}')
print(f'L2 regularisation: {model_L2.score(X_val, y_val)}')