import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

class Ridge:
    def __init__(self, lambda_=0):
        self.intercept = 0
        self.weights = None
        self.lambda_ = lambda_

    def fit_data(self, X, y):
        n_samples, n_features = X.shape

        X1 = np.hstack((np.ones((n_samples, 1)), X))
        I = np.eye(n_features + 1)

        XtX = X1.T @ X1
        XtY = X1.T @ y

        # Solving least squares problem: ŷ = (X'ᵀX' + λI) . X'ᵀy
        w_prime = np.linalg.solve(XtX + (self.lambda_) * I, XtY)

        self.intercept = w_prime[0]
        self.weights = w_prime[1:]

        return self.intercept, self.weights

    def predict(self, X):
        n_samples = X.shape[0]
        ones_vector = np.ones((n_samples,))

        # Prediction: ŷ = βᵀ * 1 + X * w
        y_pred = (self.intercept) * ones_vector + np.dot(X, self.weights)
        return y_pred


# Ridge regression for a given lambda(λ)
def ridge_regression(lambda_):
    ridge_model = Ridge(lambda_)
    β, w = ridge_model.fit_data(X_train, y_train)

    print(f"Training data with λ = {lambda_}")
    print("Intercept (β):", β)
    print("Coefficients (w):", w)

    y_pred = ridge_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}\n\n")


# Load the dataset
df = pd.read_csv("hw1/datasets/winequality-red.csv", delimiter=";")

X = df.iloc[:, :11].values
y = df.iloc[:, -1].values

# Split into training and testing sets
X_train, X_test = X[:1400], X[1400:]
y_train, y_test = y[:1400], y[1400:]

for lambda_value in [0, 0.001, 0.01, 0.1]:
    ridge_regression(lambda_value)
