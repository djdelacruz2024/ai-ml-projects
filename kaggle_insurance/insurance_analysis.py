import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('insurance.csv')

print("Shape:", df.shape)
print()
print("first 5 rows:")
print(df.head())
print()  
print("Data types:")
print(df.dtypes)
print()
print("Missing values:")
print(df.isnull().sum())
print()
print("Basic statistics:")
print(df.describe())


# Visualize charges distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['charges'], bins=50, color='steelblue', edgecolor='black')
plt.title('Distribution of Charges')
plt.xlabel('Charges')
plt.ylabel('Count')

plt.subplot(1, 3, 2)
df.boxplot(column='charges', by='smoker', ax=plt.gca())
plt.title('Charges by Smoker Status')
plt.xlabel('Smoker')
plt.ylabel('Charges')

plt.subplot(1, 3, 3)
df.boxplot(column='charges', by='region', ax=plt.gca())
plt.title('Charges by Region')
plt.xlabel('Region')
plt.ylabel('Charges')

plt.tight_layout()
plt.savefig('charges_analysis.png')
plt.show()
print("Chart saved")

# Encode categorical columns
print("\nEncoding categorical columns...")
le = LabelEncoder()
df['sex_encoded'] = le.fit_transform(df['sex'])
df['smoker_encoded'] = le.fit_transform(df['smoker'])
df['region_encoded'] = le.fit_transform(df['region'])

print("Encoded values for smoker:")
print(df[['smoker', 'smoker_encoded']].drop_duplicates())
print()
print("Encoded values for region:")
print(df[['region', 'region_encoded']].drop_duplicates())

# Prepare features for modeling
features = ['age', 'bmi', 'children', 'sex_encoded', 
            'smoker_encoded', 'region_encoded']

X = df[features]
y = df['charges']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

print("Model Comparison on Real Insurance Data:")
print("-" * 40)
print(f"Linear Regression  - R2: {lr_r2:.3f}, RMSE: ${lr_rmse:,.2f}")
print(f"Random Forest      - R2: {rf_r2:.3f}, RMSE: ${rf_rmse:,.2f}")

# Feature importance
importance = pd.Series(rf_model.feature_importances_, index=features)
importance = importance.sort_values(ascending=False)
print("\nFeature Importances:")
print(importance)
