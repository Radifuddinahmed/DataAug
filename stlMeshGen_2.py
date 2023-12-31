import numpy as np
from stl import mesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import trimesh

# Assuming you have height data in the form of a 2D matrix or a list of 2D coordinates
height_data = [ [1, 1, 0], [1, 2, 0], [1, 3, 0],[1, 4, 0],
                [2, 1, 0], [2, 2, 0], [2, 3, 0],[2, 4, 0],
                [3, 1, 0], [3, 2, 0], [3, 3, 0],[3, 4, 0],
                [4, 1, 0], [4, 2, 10], [4, 3, 10],[4, 4, 0],
                [5, 1, 0], [5, 2, 10], [5, 3, 10], [5, 4, 0],
                [6, 1, 0], [6, 2, 10], [6, 3, 10], [6, 4, 0],
                [7, 1, 0], [7, 2, 10], [7, 3, 10], [7, 4, 0],
                [8, 1, 0], [8, 2, 10], [8, 3, 10], [8, 4, 0],
                [9, 1, 0], [9, 2, 10], [9, 3, 10], [9, 4, 0],
                [10, 1, 0], [10, 2, 10], [10, 3, 10], [10, 4, 0],
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









vertices2 = np.array([[0,0],[1,0],[1,1],[0,1] ])

#[0,0,0],[1,0,0],[1,1,1],[2,1,0]
# Perform Delaunay triangulation to generate faces
tri = Delaunay(vertices2)

tri = Delaunay(vertices2)


# Create a mesh from the triangulation
mesh = trimesh.Trimesh(vertices=vertices2, faces=tri.simplices)

# Optionally, you can visualize the mesh using the show() method
mesh.show()


output_file = 'mesh.stl'
mesh.export(output_file)
