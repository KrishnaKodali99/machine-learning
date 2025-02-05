import numpy as np


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
        w_prime = np.linalg.pinv(XtX + self.lambda_ * I) @ XtY

        self.intercept = w_prime[0]
        self.weights = w_prime[1:]

        return self.intercept, self.weights

    def predict(self, X):
        n_samples = X.shape[0]
        ones_vector = np.ones((n_samples,))

        # Prediction: ŷ = βᵀ * 1 + X * w
        y_pred = (self.intercept) * ones_vector + np.dot(X, self.weights)
        return y_pred
