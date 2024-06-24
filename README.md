# pinn-burgers-pytorch
Physics-informed Neural Network (PINN) to solve the Burgers equation,




# Burgers' Equation Solver using Physics-Informed Neural Networks (PINNs)

This repository contains a Python implementation of a Physics-Informed Neural Network (PINN) for solving the Burgers' equation:

$$
\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} - \mu \frac{\partial^2 u}{\partial x^2} = 0,
$$

with a boundary condition,


$$
u(x, t=0) = -\sin(\pi x), \quad u(x=\pm1, t) = 0.
$$

The Burgers' equation is a fundamental partial differential equation from fluid mechanics and nonlinear acoustics.

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Results](#results)
- [References](#references)
- [License](#license)

## Overview

Physics-Informed Neural Networks (PINNs) incorporate physical laws into the loss function of neural networks, enabling them to solve differential equations effectively. This project applies the PINN approach to solve the Burgers' equation, demonstrating how neural networks can be used to approximate solutions to complex physical systems.



## Usage

1. **Train the Model**:
    ```python
    python main.py
    ```
    This will train the neural network to approximate the solution to the Burgers' equation based on the specified parameters in `main.py`.

2. **Visualize the Results**:
    The `show_image` function in `main.py` will generate a plot of the solution.

## Results

Upon running the model, you will see a contour plot representing the solution of the Burgers' equation over time and space. This visualization helps in understanding the dynamics of the solution and the effectiveness of the PINN approach.

### Example Plot

The example below shows the solution plot for the specified parameters:

![Burgers' Equation Solution](./sample.pdf)

## Project Structure

```
burgers-equation-pinn/
│
├── model.py              # Defines the neural network architecture
├── main.py               # Main script to solve the Burgers' equation
├── README.md             # Project documentation
└── requirements.txt      # List of dependencies
```

## References

- [Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.](https://doi.org/10.1016/j.jcp.2018.10.045)
- [Burgers' Equation - Wikipedia](https://en.wikipedia.org/wiki/Burgers'_equation)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
