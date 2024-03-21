import numpy as np
import scipy as sp
from scipy.spatial import Delaunay
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import stl
import numpy as np
from stl import mesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from stl import mesh
import trimesh


# def cube():
#     points = np.array([
#          [0, 0, 1], [0, 1, 0], [0, 1, 1],[0, 0, 0],
#         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
#     ])
#     return points
#
#
#
# points = cube()
#
#
# # Create a Delaunay triangulation of a set of points
# # points = np.array([[0, 0], [1, 0], [0, 1]])
# triangulation = Delaunay(points)
#
# # # Convert the triangulation to an STL file
# # stl_file = stl.STLFile()
# # for triangle in triangulation.simplices:
# #     vertices = points[triangle]
# #     stl_file.add_triangle(vertices)
# #
# # # Write the STL file to disk
# # with open("Delaunay.stl", "wb") as f:
# #     f.write(stl_file.tostring())
#
# indices = triangulation.simplices
# vertices = points[indices]
# faces = points[indices]
#
#
#
#
# # # Create a triangular mesh using the vertices
# # triangles = mesh.Mesh(np.zeros(len(vertices), dtype=mesh.Mesh.dtype))
# # for i, vertex in enumerate(vertices):
# #     triangles.vectors[i] = vertex
# #
# # # Write the mesh to an STL file
# # output_file = 'DelaunayTriangulation.stl'
# # triangles.save(output_file)
#
# # Convert the height data to a numpy array
# vertices = np.array(points)

# # Create a triangular mesh using the vertices
# triangles = mesh.Mesh(np.zeros(len(vertices), dtype=mesh.Mesh.dtype))
# for i, vertex in enumerate(vertices):
#     triangles.vectors[i] = vertex
#
# # Write the mesh to an STL file
# output_file = 'DelaunayTriangulation.stl'
# triangles.save(output_file)
#
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.dist = 30
# ax.azim = -140
# ax.set_xlim([0, 2])
# ax.set_ylim([0, 2])
# ax.set_zlim([0, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
#
# for f in faces:
#     face = a3.art3d.Poly3DCollection([f])
#     face.set_color(mpl.colors.rgb2hex(sp.rand(3)))
#     face.set_edgecolor('k')
#     face.set_alpha(0.5)
#     ax.add_collection3d(face)
#
# plt.show()


# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d as a3
# import numpy as np
# import scipy as sp
# from scipy import spatial as sp_spatial
#
#
#
#
#
# def cube():
#     points = np.array([
#         [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
#         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
#     ])
#     return points
#
#
#
# points = cube()
#
# hull = sp_spatial.ConvexHull(points)
# indices = hull.simplices
# faces = points[indices]
#
# print('area: ', hull.area)
# print('volume: ', hull.volume)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.dist = 30
# ax.azim = -140
# ax.set_xlim([0, 2])
# ax.set_ylim([0, 2])
# ax.set_zlim([0, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
#
# for f in faces:
#     face = a3.art3d.Poly3DCollection([f])
#     face.set_color(mpl.colors.rgb2hex(sp.rand(3)))
#     face.set_edgecolor('k')
#     face.set_alpha(0.5)
#     ax.add_collection3d(face)
#
# plt.show()

#Create Nodes
l = 15
w = 10
numberOfElements = 30

nodes = []

for x in np.linspace(0,l,num=numberOfElements):
    for y in np.linspace(0,w,num=numberOfElements):
        nodes.append([x,y])

#display the nodes
points = np.array(nodes)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
print(nodes)
# [[0.0, 0.0], 0
#  [0.0, 5.0], 1
#  [0.0, 10.0], 2
#  [7.5, 0.0], 3
#  [7.5, 5.0], 4
#  [7.5, 10.0], 5
#  [15.0, 0.0], 6
#  [15.0, 5.0], 7
#  [15.0, 10.0]] 8

#Create Elements
tri = Delaunay(points, incremental=True)
plt.triplot(points[:,0], points[:,1], tri.simplices)
plt.plot(points[:,0], points[:,1], 'o')
plt.show()
# [[3 4 0]
#  [4 1 0]
#  [4 5 1]
#  [1 5 2]
#  [7 4 3]
#  [7 3 6]
#  [7 5 4]
#  [5 7 8]]

mesh_ = tri.simplices # simplices are basically elements
print(mesh_)

# trying a different algorithm
# from pyhull.delaunay import DelaunayTri
# pyhull_t = np.asarray(DelaunayTri(points).vertices)
# print("pyhull")
# print(pyhull_t)

#Export to a file
nb_nodes = len(points) #calculate the number of nodes
nb_elements = len(mesh_) #calculate the number of elements

file = open("DelaunayTriangulation.dat", "w")
file.write("{} {} \n".format(nb_nodes,nb_elements))
for i,node in enumerate(nodes):
    file.write ("{} {} {}\n".format(i,node[0], node[1]))
file.close()

#convert to stl file
# ms = mesh.Mesh(np.zeros(mesh_.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(mesh_):
#     for j in range(3):
#         ms.vectors[i][j] = mesh_[f[j],:]
#
# ms.save('DelaunayTriangulation.stl')

# mesh.save('DelaunayTriangulation.stl')

def write_stl(vertices, faces, filename):
    with open(filename, 'w') as f:
        f.write("solid mesh\n")
        for face in faces:
            normal = np.cross(vertices[face[1]] - vertices[face[0]], vertices[face[2]] - vertices[face[0]])
            normal /= np.linalg.norm(normal)
            f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
            f.write("    outer loop\n")
            for vertex_id in face:
                vertex = vertices[vertex_id]
                f.write(f"      vertex {vertex[0]} {vertex[1]} {vertex[2]}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid mesh\n")



# Write to STL file
write_stl(points, mesh_, "DelaunayTriangulation.stl")