import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Dataset
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Score': [35,40,55,60,68,72,81,88,92,95]
}

df = pd.DataFrame(data)

# Features & Target
X = df[['Hours']]
y = df['Score']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Actual:", y_test.values)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualization
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.title("Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.show()