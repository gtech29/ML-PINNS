import torch
import torch.nn as nn
import torch.nn.functional as F

# Create a Model Class that inherits nn.Module
class Model(nn.Module):
    def __init__(self, in_features=4, h1= 8, h2= 9, out_features= 3):
        super().__init__()# instantiate nn.Module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1.h2)
        self.out = nn.Linear(h2, out_features)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        X = F.relu(self.fc2(x))
        X = self.out(x)
        
        return x
    
# Pick a manual seed for randomization
torch.manual_seed(41)
# Create an instance of the model
model = Model()