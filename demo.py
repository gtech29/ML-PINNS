import torch
#  Creating a tensor
import numpy as np
np.set_printoptions(precision=3) #This sets printing precision for NumPy array to 3 decimal places.
a = [1, 2, 3]
b = np.array([4, 5, 6], dtype=np.int32)
t_a = torch.tensor(a)
t_b = torch.from_numpy(b)
print(t_a)
print(t_b)