import numpy as np
import pandas as pd

from ridge import Ridge
from sklearn.metrics import accuracy_score


# Ridge classification for a given lambda(λ)
def ridge_classifier(lambda_):
    ridge_model = Ridge(lambda_)
    ridge_model.fit_data(X_train, y_train)

    y_pred = np.sign(ridge_model.predict(X_test))
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Training data with λ = {lambda_}")
    print(f"Accuracy: {accuracy:.4f}\n")


# Load the dataset
df = pd.read_csv("hw1/datasets/ionosphere.data", header=None)
df[34] = df[34].map({"g": 1, "b": -1})

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split into training and testing sets
X_train, X_test = X[:300], X[300:]
y_train, y_test = y[:300], y[300:]

# Training and evaluation
for lambda_ in [0, 0.001, 0.01, 0.1]:
    ridge_classifier(lambda_)
