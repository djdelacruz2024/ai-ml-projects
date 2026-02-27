import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n_customers = 1000
data = {
    'age': np.random.randint(18,  70, n_customers),
    'monthly_charges': np.random.uniform(20, 100, n_customers),
    'tenure_months': np.random.randint(1, 72, n_customers),
    'num_products': np.random.randint(1, 5, n_customers),
    'num_supportcalls': np.random.randint(0, 10, n_customers),
}
df = pd.DataFrame(data)
df['churn'] = ((df['num_supportcalls'] > 5) | (df['monthly_charges'] > 80) & (df['tenure_months'] < 12)).astype(int)

print(df.head())
print(df.shape)
print()
print("Churn distribution:")
print(df['churn'].value_counts())

x = df.drop('churn', axis=1)
y = df['churn']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print("Before scaling - first row of x_train:")
print(x_train.iloc[0].values)
print()
print("After scaling - first row of x_train_scaled:")
print(x_train_scaled[0])

model = LogisticRegression(random_state = 42)
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['stayed', 'churned']))