
# Manipulating the data type and shape of a tensor

There are many methods in PyTorch to change the shape and data type of a tensor. Changing the data type and shape can impact memory usage and performance, so it's important to choose the appropriate method based on the requirements of the model and available resources.

This section explains different methods to manipulate tensors, mainly using PyTorch functions that cast, reshape, transpose, and squeeze (remove dimensions).

## Transposing a Tensor

1. Create a tensor 't' with shape (3,5) filled with random numbers.
2. Apply the transpose method to 't'. By doing this, the dimensions of the tensor are modified based on the provided parameters.

   - The first parameter is the tensor to be transposed, and the second and third parameters specify the dimensions that are to be swapped. In this case, the first dimension (rows) will be swapped with the second dimension (columns).

## Reshaping a tensor (for example, from a 1D vector to a 2D array):

1. Create a tensor `t` filled with 30 zeros using `torch.zeros()`. The size of this tensor is 1D with 30 elements.
2. Apply the `reshape()` method to `t`, specifying that it should be reshaped into a 2D tensor with 5 rows and 6 columns. This operation effectively transforms the 1D vector into a 2D array.
3. Print the shape of the reshaped tensor `t_reshape`, showing that it indeed has dimensions `[5, 6]`.

## Removing the unnecessary dimensions (dimensions that have size 1, which are not needed):

1. Create a tensor 't' filled with zeros. It has layers of dimensions: layer 1 has 2 more dimensions, and inside layer 2, there is 1 more dimension with 4 rows and 1 column filled with zeros.
2. Apply the squeeze method with a parameter 2, indicating the dimension on which to squeeze the tensor. Since the 3rd dimension has a size 1, it gets removed, resulting in the tensor with dimension (1,2,4,1).

Squeezing tensors is useful for removing unnecessary dimensions, which can improve performance without losing any important information.
