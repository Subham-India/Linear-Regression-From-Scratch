import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0
        
        for i in range(self.iterations):
            y_pred = np.dot(X, self.coefficients) + self.intercept
            errors = y_pred - y
            
            dw = (2 / n_samples) * np.dot(X.T, errors)
            db = (2 / n_samples) * np.sum(errors)
            
            self.coefficients = self.coefficients - self.learning_rate * dw
            self.intercept = self.intercept - self.learning_rate * db
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept


# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2.5 * X.squeeze() + 3 + np.random.randn(100) * 2

# Train model
model = LinearRegression(learning_rate=0.01, iterations=1000)
model.fit(X, y)

# Predict
y_pred = model.predict(X)

# Plot
plt.scatter(X, y, alpha=0.6)
plt.plot(X, y_pred, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"Intercept: {model.intercept}")
print(f"Coefficient: {model.coefficients[0]}")
