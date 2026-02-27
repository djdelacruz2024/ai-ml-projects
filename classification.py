import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
df["target"] = cancer.target

print(df.head())
print(df.shape)
print()
print("Target values:", cancer.target_names)
print("Class distribution: ")
print(df["target"].value_counts())

x = df.drop("target", axis=1)
y = df["target"]    

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

print("Model trained successfully.")
print(f"traning samples: {x_train.shape[0]}")
print(f"Testing samples: {x_test.shape[0]}")

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))