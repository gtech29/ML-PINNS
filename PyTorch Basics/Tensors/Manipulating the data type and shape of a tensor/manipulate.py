# Import necessary libraries
import torch


# Transposing a tensor:
t = torch.rand(3, 5)
t_tr = torch.transpose(t, 0, 1)
print(t.shape, ' --> ', t_tr.shape,"\n")


# Reshaping a tensor (for example, from a 1D vector to a 2D array):
t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print(t_reshape.shape, "\n")
torch.Size([5, 6])


# Removing the unnecessary dimensions (dimensions that have size 1, which are not needed):
t = torch.zeros(1, 2, 1, 4, 1)
t_sqz = torch.squeeze(t, 2)
print(t.shape, " --> ", t_sqz.shape, "\n")
