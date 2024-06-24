import torch
import numpy as np
import matplotlib.pyplot as plt

def show_image(model: torch.nn.Module, x_max: float, x_min: float, t_max: float):
    """
    Visualizes the solution of the Burgers' equation using a trained model.

    Args:
        model (torch.nn.Module): The trained neural network model.
        x_max (float): The maximum value of the spatial domain.
        x_min (float): The minimum value of the spatial domain.
        t_max (float): The maximum value of the temporal domain.
    """
    x = torch.linspace(x_min, x_max, steps=100)
    t = torch.linspace(0.0, t_max, steps=100)
    x, t = torch.meshgrid(x, t, indexing='xy')
    u = model(torch.stack([x.flatten(), t.flatten()], dim=-1)).view(100, 100)

    plt.figure(figsize=(8, 4))
    plt.contourf(t.detach().numpy(), x.detach().numpy(), u.detach().numpy(), 100, cmap='jet')
    plt.colorbar()

    plt.xlabel("t")
    plt.ylabel("x")
    plt.show()


def solve_burgers(
        model: torch.nn.Module,
        nu: float,
        x_max: float,
        x_min: float,
        t_max: float,
        num_bc_init: int,
        num_bc_max: int,
        num_bc_min: int,
        num_collocation: int,
        device: str,
        num_epoch: int,
    ):
    """
    Solves the Burgers' equation using a Physics-Informed Neural Network (PINN).

    Args:
        model (torch.nn.Module): The neural network model.
        nu (float): The viscosity coefficient.
        x_max (float): The maximum value of the spatial domain.
        x_min (float): The minimum value of the spatial domain.
        t_max (float): The maximum value of the temporal domain.
        num_bc_init (int): Number of initial boundary condition points.
        num_bc_max (int): Number of boundary condition points at x = x_max.
        num_bc_min (int): Number of boundary condition points at x = x_min.
        num_collocation (int): Number of collocation points.
        device (str): Device to run the computations on ('cpu' or 'cuda').
        num_epoch (int): Number of training epochs.
    """
    # Boundary condition at x = x_max
    xt_max = torch.rand(num_bc_max, 2) * t_max  # Random points in time
    xt_max[:, 0] = x_max  # Set spatial coordinate to x_max
    u_max = torch.zeros(num_bc_max)  # Velocity at boundary is zero

    # Boundary condition at x = x_min
    xt_min = torch.rand(num_bc_min, 2) * t_max
    xt_min[:, 0] = x_min
    u_min = torch.zeros(num_bc_min)

    # Initial condition (t = 0)
    xt_init = torch.rand(num_bc_init, 2) * (x_max - x_min) + x_min
    xt_init[:, 1] = 0.0 # Set temporal coordinate to 0
    u_init = -torch.sin(np.pi * xt_init[:, 0])

    # Collocation points
    data = torch.rand(num_collocation, 2)
    data[:, 0] = data[:, 0] * (x_max - x_min) + x_min
    data[:, 1] = data[:, 1] * t_max

    # Merge points
    num_cond = num_bc_init + num_bc_max + num_bc_min
    xt = torch.cat([xt_max, xt_min, xt_init, data], dim=0)
    xt.requires_grad = True
    u_cond = torch.cat([u_max, u_min, u_init], dim=0)[:, None]

    # Train the model
    model.to(device)
    model.train()
    xt.to(device)

    for epoch in range(num_epoch):
        optimizer.zero_grad()
        u = model(xt)
        u_x_t = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x_t[:, 0], xt, grad_outputs=torch.ones_like(u_x_t[:, 0]), create_graph=True)[0]

        # Differential equation residual
        loss_diff = torch.mean((u_x_t[num_cond:, 1] + u[num_cond:, 0] * u_x_t[num_cond:, 0] - nu * u_xx[num_cond:, 0]) ** 2)
        loss_cond = torch.mean((u[:num_cond, :] - u_cond) ** 2)
        loss = loss_diff + loss_cond

        # Boundary and initial condition loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(epoch, loss.item())


if __name__ == "__main__":
    from model import MLP  # Import a custom MLP model
    model = MLP(in_dim=2, num_mid_layers=8, mid_dim=32, out_dim=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

    solve_burgers(
        model=model,
        nu=0.01/np.pi,
        x_max=1.0,
        x_min=-1.0,
        t_max=1.0,
        num_bc_init=100,
        num_bc_max=100,
        num_bc_min=100,
        num_collocation=5000,
        device="cpu",
        num_epoch=5000,
    )

    show_image(model=model, x_max=1.0, x_min=-1.0, t_max=1.0)
    
