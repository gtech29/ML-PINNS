import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()  # instantiate nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1.h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        X = F.relu(self.fc2(x))
        X = self.out(x)

        return x


# Pick a manual seed for randomization
torch.manual_seed(41)
# Create an instance of the model
model = Model()

# Load the data
url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
my_df = pd.read_csv(url)

# Data cleansing
my_df["species"] = my_df["species"].replace("setosa", 0.0)
my_df["species"] = my_df["species"].replace("versicolor", 1.0)
my_df["species"] = my_df["species"].replace("virginica", 2.0)

# Train the model
X = my_df.drop("species", axis=1)
y = my_df["species"]

# Convert to numpy arrays
X = X.values
y = y.values

# train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=41)

# Convert X features to float tensors
X_train= torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y labels to tensor long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Set criterion of model to measure error, how far off predictions are from data
criterion = nn.CrossEntropyLoss()
# Choose Optimizer (Adam), set learning rate ( if error doesnt go down after a bunch of epochs, we might lower learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

