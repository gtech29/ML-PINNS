# Import necessary libraries
import torch
import numpy as np

# Creating a tensor from a list
np.set_printoptions(precision=3)
a = [1, 2, 3]
b = np.array([4, 5, 6], dtype=np.int32)
t_a = torch.tensor(a)
t_b = torch.from_numpy(b)
print(t_a)
print(t_b,"\n")

# Tensor with a specific shape
t_ones = torch.ones(2, 3)
t_ones.shape 
torch.Size([2, 3])
print(t_ones,"\n")

# Create a tensor with random values:
rand_tensor = torch.rand(2, 3)
print(rand_tensor, "\n")
