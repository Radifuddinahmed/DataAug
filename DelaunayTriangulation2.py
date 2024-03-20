import numpy as np
from stl import mesh
import numpy as np
from stl import mesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from stl import mesh
from random import randint

vertices = np.array([
    [0, 0, 1],  # 0
    [0, 1, 1],  # 1
    [1, 1, 1],  # 2
    [1, 0, 1],  # 3
    [0, 0, 0],  # 4
    [0, 1, 0],  # 5
    [1, 1, 0],  # 6
    [1, 0, 0],  # 7
    [3, 3, 0],  # 8
    [3, 2, 0],  # 9
    [0, 1, -3],  # 10
    [3, 3, -3],  # 11
    [4, 2, 0],  # 12
    [0, 0, -2],  # 13
    [1, 0, -2],  # 14
    [0, 0, -3],  # 15
    [3, 2, -3],  # 16
    [3, 2, -2],  # 17
    [4, 2, -2],  # 18
])

faces = np.array([
    [0, 2, 1],
    [0, 3, 2],

    [0, 1, 5],
    [0, 5, 4],

    [5, 1, 2],
    [5, 2, 6],

    [3, 6, 2],
    [3, 7, 6],

    [0, 4, 7],
    [0, 7, 3],

    [6, 8, 5],
    [6, 9, 8],

    [5, 8, 11],
    [5, 11, 10],

    [6, 7, 12],
    [6, 12, 9],

    [4, 13, 14],
    [4, 14, 7],

    [4, 10, 15],
    [4, 5, 10],

    [8, 16, 11],
    [8, 9, 16],

    [9, 12, 18],
    [9, 18, 17],

    [7, 18, 12],
    [7, 14, 18],

    [13, 17, 18],
    [13, 18, 14],

    [13, 15, 16],
    [13, 16, 17],

    [15, 10, 11],
    [15, 11, 16],
])

shape = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        shape.vectors[i][j] = vertices[f[j], :]

shape.save("DelaunayTriangulation2.stl")

# #Create Nodes
# l = 15
# w = 10
# h = 2
# numberOfElements = 10
#
# nodes = []
#
# for x in np.linspace(0,l,num=numberOfElements):
#     for y in np.linspace(0,w,num=numberOfElements):
#
#         nodes.append([x,y,randint(1, 5)])
#
# #display the nodes
# points = np.array(nodes)
# plt.plot(points[:,0], points[:,1], 'o')
# plt.show()
#
# #Create Elements
# tri = Delaunay(points)
# # plt.triplot(points[:,0], points[:,1], tri.simplices)
# # plt.plot(points[:,0], points[:,1], 'o')
# # plt.show()
#
# mesh_ = tri.simplices
# print(mesh_)
#
#
# #Export to a file
# nb_nodes = len(points) #calculate the number of nodes
# nb_elements = len(mesh_) #calculate the number of elements
#
# file = open("DelaunayTriangulation.dat", "w")
# file.write("{} {} \n".format(nb_nodes,nb_elements))
# for i,node in enumerate(nodes):
#     file.write ("{} {} {}\n".format(i,node[0], node[1]))
# file.close()
#
# #convert to stl file
#
#
# ms = mesh.Mesh(np.zeros(mesh_.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(mesh_):
#     for j in range(3):
#         ms.vectors[i][j] = mesh_[f[j],:]
#
# ms.save('DelaunayTriangulation2.stl')