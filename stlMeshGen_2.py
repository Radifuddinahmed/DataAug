import numpy as np
from stl import mesh

# Assuming you have height data in the form of a 2D matrix or a list of 2D coordinates
height_data = [ [1, 1, 0], [1, 2, 0], [1, 3, 0],[1, 4, 0],
                [2, 1, 0], [2, 2, 0], [2, 3, 0],[2, 4, 0],
                [3, 1, 0], [3, 2, 0], [3, 3, 0],[3, 4, 0],
                [4, 1, 0], [4, 2, 1], [4, 3, 1],[4, 4, 1],
                [5, 1, 0], [5, 2, 1], [5, 3, 1], [5, 4, 1],
                [6, 1, 0], [6, 2, 1], [6, 3, 1], [6, 4, 1],
                [7, 1, 0], [7, 2, 1], [7, 3, 1], [7, 4, 1],
                [8, 1, 0], [8, 2, 1], [8, 3, 1], [8, 4, 1],
                [9, 1, 0], [9, 2, 1], [9, 3, 1], [9, 4, 1],
                [10, 1, 0], [10, 2, 1], [10, 3, 1], [10, 4, 1],
                [11, 1, 0], [11, 2, 0], [11, 3, 0], [11, 4, 0],
                [12, 1, 0], [12, 2, 0], [12, 3, 0], [12, 4, 0],
                [13, 1, 0], [13, 2, 0], [13, 3, 0], [13, 4, 0],
]
# Convert the height data to a numpy array
vertices = np.array(height_data)

# Create a triangular mesh using the vertices
triangles = mesh.Mesh(np.zeros(len(vertices), dtype=mesh.Mesh.dtype))
for i, vertex in enumerate(vertices):
    triangles.vectors[i] = vertex

# Write the mesh to an STL file
output_file = 'output.stl'
triangles.save(output_file)