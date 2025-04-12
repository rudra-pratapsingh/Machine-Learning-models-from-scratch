import numpy as np
 
class SVM_model:
  def __init__(self, lambda_param=0.01, n_iters=1000, lr=0.001):
    self.lr = lr
    self.lambda_param = lambda_param
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  def fit(self, X, y):
    n_samples, n_feautres = X.shape

    yi = np.where(y<=0, -1, 1)

    self.weights = np.zeros(n_feautres)
    self.bias = 0

    for _ in range (self.n_iters):
        for idx, Xi in enumerate(X):
          condition = yi[idx] * (np.dot(Xi, self.weights) - self.bias) >=1

          if condition:
            dw = 2 * self.lambda_param * self.weights
            self.weights -= self.lr * dw

          else:
            dw = 2 * self.lambda_param * self.weights - np.dot(Xi, yi[idx])
            db = yi[idx]

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

  def predict(self, X):
    approx = np.dot(X, self.weights) - self.bias
    return np.sign(approx)