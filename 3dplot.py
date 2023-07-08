import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Generate example data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')


# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot')

# Add color map
ax.set_zlim(-1, 1)  # Set the z-axis limits
fig.colorbar(fig, shrink=0.5, aspect=5)  # Add color bar

plt.show()
