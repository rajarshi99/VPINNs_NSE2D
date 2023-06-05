import numpy as np
import matplotlib.pyplot as plt

def solve_poisson_2d_sor(n, L=1.0, omega=1.8):
    # Create grid
    x = np.linspace(0, L, n)
    y = np.linspace(0, L, n)
    X, Y = np.meshgrid(x, y)

    return np.sin(np.pi * X) * np.cos(np.pi * Y) * Y * (1-Y)


# Solve the Poisson problem using SOR method
n = 100  # Number of grid points in each dimension
solution_sor = solve_poisson_2d_sor(n, omega=1.8)

# Plot the contourf plot of the solution
plt.contourf(solution_sor, cmap='jet')
plt.colorbar()

# Save the plot as solution.png
plt.savefig('solution.png')
plt.show()
