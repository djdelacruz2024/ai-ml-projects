import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


california = fetch_california_housing()

df = pd.DataFrame(california.data, columns=california.feature_names)

df['PRICE'] = california.target

print(df.head())

print(df.shape )

print(df.isnull().sum())

X = df.drop("PRICE", axis=1)
y = df["PRICE"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)

model = LinearRegression()
model.fit(x_train, y_train)

print("Model trained successfully.")

y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = model.score(x_test, y_test)

print(f"RSME: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.plot([0, 5], [0, 5], 'r--') 
plt.show()
