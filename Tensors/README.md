# **Ways of creating a tensor**

In this module, I wrote down the different ways of creating a sensor that are discussed in the textbook provided by Dr. Peng.

* **Creating a tensor from a list:**

  * First, import the necessary libraries, in this case PyTorch and NumPy.
  * Next, set the printing precision of the NumPy array to 3 decimal places.
  * Create a list for the first tensor.
  * Create an array for the second tensor, in here it is specified that the data type of the elements should be 32-bit integers.
  * Lastly, convert the list and numpy array into tensors and print.

    * Note:
      * **`t_b = torch.from_numpy(b)`** : This converts the NumPy array `b` into a PyTorch tensor `t_b`. The `from_numpy` function creates a tensor that shares memory with the original NumPy array, meaning changes to the tensor will also affect the NumPy array and vice versa
