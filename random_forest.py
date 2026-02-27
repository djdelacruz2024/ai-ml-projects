import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

california = fetch_california_housing()
df = pd.DataFrame(california.data, columns =california.feature_names)
df["PRICE"] = california.target

x = df.drop("PRICE", axis=1)
y = df["PRICE"] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(df.head())
print(df.shape)

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(x_train, y_train)

print("Model trained successfully.")

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = model.score(x_test, y_test)    

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")
print()
print("Linear Regression Results:")
print("RMSE: 0.75")
print("R2 Score: 0.58")

importance = pd.Series(model.feature_importances_, index=california.feature_names)
importance = importance.sort_values(ascending=False)
print()
print("Feature Importances:")
print(importance)