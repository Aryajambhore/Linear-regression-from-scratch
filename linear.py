import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and clean data
df = pd.read_csv("CAR_DETAILS_FROM_CAR_DEKHO.csv")
df.dropna(inplace=True)

# Feature and target
X_raw = df["year"].values
y = df["selling_price"].values

# Normalize X
X = (X_raw - np.mean(X_raw)) / np.std(X_raw)

# Hyperparameters
m = 0
b = 0
learning_rate = 0.01
epochs = 2000
n = len(X)

# Gradient descent loop
for i in range(epochs):
    y_pred = m * X + b
    error = y - y_pred

    dm = (-2 / n) * np.sum(X * error)
    db = (-2 / n) * np.sum(error)

    m -= learning_rate * dm
    b -= learning_rate * db

    if i % 100 == 0:
        mse = np.mean(error ** 2)
        print(f"Epoch {i}, MSE: {mse:.2f}")

print(f"\nFinal Model: y = {m:.2f}x + {b:.2f}")
print(df["year"].corr(df["selling_price"]))

# Plotting
plt.scatter(X, y, color='blue', label="Actual Data")
plt.plot(X, m * X + b, color='red', label="Regression Line")
plt.xlabel("year")
plt.ylabel("selling_price")
plt.title("Linear Regression on Car Data")
plt.legend()
plt.grid(True)
plt.show()
