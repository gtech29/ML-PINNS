import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Define the MLP model with increased complexity
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x


# Differential equation y'' + y = 0
def differential_equation(x, y, dy_dx, d2y_dx2):
    return d2y_dx2 + y


# Boundary conditions: y(0) = 1, y'(0) = 0
def boundary_condition(x, y, dy_dx):
    y0 = y[0]
    dy_dx0 = dy_dx[0]
    return y0 - 1, dy_dx0


# Compute the derivatives using automatic differentiation
def compute_derivatives(model, x):
    y = model(x)
    dy_dx = torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]
    d2y_dx2 = torch.autograd.grad(
        dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True
    )[0]
    return y, dy_dx, d2y_dx2


# Loss function
def loss_function(model, x_interior, x_boundary):
    y_interior, dy_dx_interior, d2y_dx2_interior = compute_derivatives(
        model, x_interior
    )
    y_boundary, dy_dx_boundary, _ = compute_derivatives(model, x_boundary)

    # Differential equation loss
    loss_de = torch.mean(
        (
            differential_equation(
                x_interior, y_interior, dy_dx_interior, d2y_dx2_interior
            )
        )
        ** 2
    )

    # Boundary condition loss
    loss_bc_y, loss_bc_dy_dx = boundary_condition(
        x_boundary, y_boundary, dy_dx_boundary
    )
    loss_bc = torch.mean(loss_bc_y**2) + torch.mean(loss_bc_dy_dx**2)

    # Total loss
    loss = loss_de + loss_bc
    return loss


# Training function
def train(model, optimizer, x_interior, x_boundary, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(model, x_interior, x_boundary)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


# Define the domain
x_interior = torch.linspace(0, 2 * np.pi, 100, requires_grad=True).view(-1, 1)
x_boundary = torch.tensor([[0.0]], requires_grad=True)

# Initialize the model, optimizer, and train
input_dim = 1
hidden_dim = 20  # Increased number of hidden units
output_dim = 1
model = MLP(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(
    model.parameters(), lr=0.001, weight_decay=1e-5
)  # Added weight decay

# Train the model
epochs = 5000  # Increased number of epochs
train(model, optimizer, x_interior, x_boundary, epochs)

# Plot the solution
model.eval()
x_test = torch.linspace(0, 2 * np.pi, 100).view(-1, 1)
y_test = model(x_test).detach().numpy()
plt.plot(x_test.numpy(), y_test, label="MLP solution")
plt.plot(x_test.numpy(), np.cos(x_test.numpy()), label="Analytic solution")
plt.legend()
plt.show()
