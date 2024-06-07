import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()  # instantiate nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# Pick a manual seed for randomization
torch.manual_seed(41)

# Create an instance of the model
model = Model()

# Load the data
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_df = pd.read_csv(url)

# Data cleansing
my_df["species"] = my_df["species"].replace(
    {"setosa": 0.0, "versicolor": 1.0, "virginica": 2.0}
)
my_df["species"] = my_df["species"].astype(int)

# Train the model
X = my_df.drop("species", axis=1)
y = my_df["species"]

# Convert to numpy arrays
X = X.values
y = y.values

# Ensure y is of the correct type
y = y.astype(int)

# Train, test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=41
)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensor long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set criterion of model to measure error, how far off predictions are from data
criterion = nn.CrossEntropyLoss()

# Choose Optimizer (Adam), set learning rate (if error doesn't go down after a bunch of epochs, we might lower learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop (example)
epochs = 100
losses = []

for epoch in range(epochs):
    optimizer.zero_grad()  # Zero the gradients
    y_pred = model(X_train)  # Forward pass
    loss = criterion(y_pred, y_train)  # Compute the loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update the weights

    losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Plotting the training loss
plt.plot(range(epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
