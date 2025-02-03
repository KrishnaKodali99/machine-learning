import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Load the dataset
df = pd.read_csv("hw1/datasets/ionosphere.data", header=None)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
y = np.where(y == 'g', 1, -1)

# Add constant term (bias) to features
X = np.hstack((X, np.ones((X.shape[0], 1))))

# Split into training and testing sets
X_train, X_test = X[:300], X[300:]
y_train, y_test = y[:300], y[300:]

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training and evaluation
for lambda_val in [0, 0.001, 0.01, 0.1]:
    model = Ridge(alpha=lambda_val, fit_intercept=False)
    model.fit(X_train, y_train)
    
    y_pred = np.sign(model.predict(X_test))
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Lambda: {lambda_val}, Accuracy: {accuracy:.2f}%")
