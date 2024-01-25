# pinn-burgers-pytorch
Physics-informed Neural Network (PINN) to solve the Burgers equation,


$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \mu \frac{\partial^2 u}{\partial x^2} = 0,
$$

with a boundary condition,


$$
u(x, t=0) = -\sin(\pi x), \quad u(x=\pm1, t) = 0.
$$
