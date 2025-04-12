import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn.datasets import make_regression

X,y = make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], y, color = "b", marker = "o", s = 30)
plt.show()

model = LinearRegression(lr = 1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

def mse(y_test, y_pred):
  return np.mean((y_test-y_pred)**2)

mse = mse(y_test, y_pred)
print("Mean Squared Error: ", mse)

y_pred_line = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='red', linewidth=2, label='Prediction')
plt.show()