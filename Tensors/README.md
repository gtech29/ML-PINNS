# Ways of creating a tensor

## Creating a tensor from a list:

1. First, import the necessary libraries, in this case PyTorch and NumPy.
2. Next, set the printing precision of the NumPy array to 3 decimal places.
3. Create a list for the first tensor.
4. Create an array for the second tensor, in here it is specified that the data type of the elements should be 32-bit integers.
5. Lastly, convert the list and numpy array into tensors and print.

   * **Note:**
     * **`t_b = torch.from_numpy(b)`**: This converts the NumPy array `b` into a PyTorch tensor `t_b`. The `from_numpy` function creates a tensor that shares memory with the original NumPy array, meaning changes to the tensor will also affect the NumPy array and vice versa.

## Create a tensor with a specific shape:

1. Generate a tensor with dimensions of 2 rows and 3 columns, each element initialized to the value 1.
2. Retrieve and print the shape of the created tensor.

## Create a tensor with random values:

### `torch.rand(2,3)`:
   * This function call generates a tensor of random values with a shape of `(2, 3)`. The `torch.rand()` function creates a tensor of the specified shape (in this case, 2 rows and 3 columns) and fills it with random numbers sampled from a uniform distribution between 0 and 1.

### rand_tensor = torch.rand(2,3):
   * The result of the `torch.rand(2,3)` call is assigned to the variable `rand_tensor`. This variable now holds the randomly generated tensor.

### `print(rand_tensor)`:
   * Finally, the code prints the randomly generated tensor. The output will display the tensor with its values. In this case, it's a 2x3 tensor with random values between 0 and 1.
