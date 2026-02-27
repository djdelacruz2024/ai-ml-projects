import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

california = fetch_california_housing()
x = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train_tensor = torch.FloatTensor(x_train_scaled)
x_test_tensor = torch.FloatTensor(x_test_scaled)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

print("Data prepared successfully")
print(f"Traing shape: {x_train_tensor.shape}")
print(f"Testing shape: {x_test_tensor.shape}")

# Define the neural network


class HousingModel(nn.Module):
    def __init__(self):
        super(HousingModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Create the model
model = HousingModel()
print(model)


# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    predictions = model(x_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    test_predictions = model(x_test_tensor)
    test_loss = criterion(test_predictions, y_test_tensor)
    rmse = torch.sqrt(test_loss)

print(f"Test RMSE: {rmse.item():.4f}")
print()
print("Model Comparison:")
print("Linear Regression RMSE:  0.75")
print("Random Forest RMSE:      0.51")
print(f"Neural Network RMSE:     {rmse.item():.2f}")
