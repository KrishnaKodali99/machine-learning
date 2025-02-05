import pandas as pd

from ridge import Ridge
from sklearn.metrics import mean_squared_error


# Ridge regression for a given lambda(λ)
def ridge_regression(lambda_):
    ridge_model = Ridge(lambda_)
    ridge_model.fit_data(X_train, y_train)

    y_pred = ridge_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    print(f"Training data with λ = {lambda_}")
    print(f"Mean Squared Error: {mse}\n")


# Load the dataset
df = pd.read_csv("hw1/datasets/winequality-red.csv", delimiter=";")

X = df.iloc[:, :11].values
y = df.iloc[:, -1].values

# Split into training and testing sets
X_train, X_test = X[:1400], X[1400:]
y_train, y_test = y[:1400], y[1400:]

for lambda_value in [0, 0.001, 0.01, 0.1]:
    ridge_regression(lambda_value)
