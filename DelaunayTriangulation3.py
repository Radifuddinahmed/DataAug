import numpy as np
from stl import mesh
import numpy as np
from stl import mesh
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from stl import mesh
from random import randint
import open3d as o3d
import trimesh

# vertices = np.array([
#     [0, 0, 1],  # 0
#     [0, 1, 1],  # 1
#     [1, 1, 1],  # 2
#     [1, 0, 1],  # 3
#     [0, 0, 0],  # 4
#     [0, 1, 0],  # 5
#     [1, 1, 0],  # 6
#     [1, 0, 0],  # 7
#     [3, 3, 0],  # 8
#     [3, 2, 0],  # 9
#     [0, 1, -3],  # 10
#     [3, 3, -3],  # 11
#     [4, 2, 0],  # 12
#     [0, 0, -2],  # 13
#     [1, 0, -2],  # 14
#     [0, 0, -3],  # 15
#     [3, 2, -3],  # 16
#     [3, 2, -2],  # 17
#     [4, 2, -2],  # 18
# ])

# faces = np.array([
#     [0, 2, 1],
#     [0, 3, 2],
#
#     [0, 1, 5],
#     [0, 5, 4],
#
#     [5, 1, 2],
#     [5, 2, 6],
#
#     [3, 6, 2],
#     [3, 7, 6],
#
#     [0, 4, 7],
#     [0, 7, 3],
#
#     [6, 8, 5],
#     [6, 9, 8],
#
#     [5, 8, 11],
#     [5, 11, 10],
#
#     [6, 7, 12],
#     [6, 12, 9],
#
#     [4, 13, 14],
#     [4, 14, 7],
#
#     [4, 10, 15],
#     [4, 5, 10],
#
#     [8, 16, 11],
#     [8, 9, 16],
#
#     [9, 12, 18],
#     [9, 18, 17],
#
#     [7, 18, 12],
#     [7, 14, 18],
#
#     [13, 17, 18],
#     [13, 18, 14],
#
#     [13, 15, 16],
#     [13, 16, 17],
#
#     [15, 10, 11],
#     [15, 11, 16],
# ])

# shape = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces):
#     for j in range(3):
#         shape.vectors[i][j] = vertices[f[j], :]
#
# shape.save("DelaunayTriangulation2.stl")
#
# #write vertices as pcd file
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(vertices)
# o3d.io.write_point_cloud("DelaunayTriangulation3.pcd", pcd)
# o3d.visualization.draw_geometries([pcd])


# print("Testing IO for point cloud ...")
# sample_pcd_data = o3d.data.PCDPointCloud()
# pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
# xyz = o3d.io.read_xyz()
#
# print(pcd)
# o3d.io.write_point_cloud("copy_of_fragment.stl", stl)


# pcd = o3d.io.read_point_cloud("DelaunayTriangulation3.pcd")
# pcd.estimate_normals()

# estimate radius for rolling ball
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 1.5 * avg_dist
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#            pcd,
#            o3d.utility.DoubleVector([radius, radius * 2]))
#
# # create the triangular mesh with the vertices and faces from open3d
# tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
#                           vertex_normals=np.asarray(mesh.vertex_normals))
#
# trimesh.convex.is_convex(tri_mesh)
#
# trimesh.exchange.export.export_mesh(tri_mesh, 'DelaunayTriangulation3.stl')

#new mesh data

# knot_mesh = o3d.data.KnotMesh()
# mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
# mesh.compute_vertex_normals()
# print(
#     f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
# )
# o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True)
# mesh = mesh.subdivide_loop(number_of_iterations=1)
# print(
#     f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
# )
# o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True)


#new mesh data

# pcd = o3d.io.read_point_cloud('No title.pcd')
# pcd.estimate_normals()
#
# # to obtain a consistent normal orientation
# pcd.orient_normals_towards_camera_location(pcd.get_center())
#
# # or you might want to flip the normals to make them point outward, not mandatory
# pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))
#
# # surface reconstruction using Poisson reconstruction
# mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
#
# # paint uniform color to better visualize, not mandatory
# mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))
#
# o3d.io.write_triangle_mesh('DelaunayTriangulation3_mesh.ply', mesh)



# vertices = np.array([
#         [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
#         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
#     ])
#
# # tri = Delaunay(vertices, incremental=True)
# # plt.triplot(vertices[:,0], vertices[:,1], tri.simplices)
# # plt.plot(vertices[:,0], vertices[:,1], 'o')
# # plt.show()
#
# #write vertices as pcd file
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(vertices)
# o3d.io.write_point_cloud("DelaunayTriangulation4.pcd", pcd)
# o3d.visualization.draw_geometries([pcd])
#
# #read the vertices as pcd file
# # pcd = o3d.io.read_point_cloud("DelaunayTriangulation3.pcd")
# pcd = o3d.io.read_point_cloud("sphere.ply")
# pcd.estimate_normals()
#
#
# # Create a point cloud
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(vertices)
#
# # Estimate the normals of the point cloud
# pcd.estimate_normals()
#
# # estimate radius for rolling ball
# distances = pcd.compute_nearest_neighbor_distance()
# avg_dist = np.mean(distances)
# radius = 1.5 * avg_dist
#
# radii = [20]
#
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#            pcd,
#            o3d.utility.DoubleVector(radii))
#
#
# # Visualize the triangle mesh
# o3d.visualization.draw_geometries([mesh])






######### alpha
# Read and visualize point cloud data
pcd = o3d.io.read_point_cloud("SensorPcdDataGenerator20240317181729.pcd")
o3d.visualization.draw_geometries([pcd])
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)
# the lines above created a tangent based normal, before using this the normals were not tangent
# hence the surface of the sphere were not filled
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

########## mesh creation and visualization using the "alpha shapes method"
# alpha = 0.2
#     #0.03 has holes
#     #0.09 has holes
#     #0.1 has holes
#     #0.5 shows best results with full surface
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


######### mesh creation and visualization using the "alpha shapes method"
# radii = [0.02]
# #0.005!not filled, 0.01!partially filled, 0.02!partially filled, 0.04!partially filled
# #.2 .3 .4
# # previous values did not work due to surfaces being partially normal
# ## now with the new tangent based normal algorithm it is working now
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([pcd, mesh])



######### mesh creation and visualization using the "Poisson Surface Reconstruction"
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=1.4, # zoom value
#                                   front=[-0.4761, -0.1698, -0.4434],
#                                   lookat=[1.05, 0.8, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])
#
# print('run Poisson surface reconstruction')
# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
#         pcd, depth=9) # tried values from .1 to 14, nothing worked
# print(mesh)
# o3d.visualization.draw_geometries([mesh],
#                                   zoom=1.4, # zoom value
#                                   front=[-0.4761, -0.1698, -0.4434],
#                                   lookat=[1.05, 0.8, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])
#
# print('visualize densities')
# densities = np.asarray(densities)
# density_colors = plt.get_cmap('plasma')(
#     (densities - densities.min()) / (densities.max() - densities.min()))
# density_colors = density_colors[:, :3]
# density_mesh = o3d.geometry.TriangleMesh()
# density_mesh.vertices = mesh.vertices
# density_mesh.triangles = mesh.triangles
# density_mesh.triangle_normals = mesh.triangle_normals
# density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
# o3d.visualization.draw_geometries([density_mesh],
#                                   zoom=1.4, # zoom value
#                                   front=[-0.4761, -0.1698, -0.4434],
#                                   lookat=[1.05, 0.8, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])
#
# print('remove low density vertices')
# vertices_to_remove = densities < np.quantile(densities, 0.01)
# mesh.remove_vertices_by_mask(vertices_to_remove)
# print(mesh)
# o3d.visualization.draw_geometries([mesh],
#                                   zoom=1.4, # zoom value
#                                   front=[-0.4761, -0.1698, -0.4434],
#                                   lookat=[1.05, 0.8, 0.9284],
#                                   up=[0.2304, -0.8825, 0.4101])

########## Normal Estimation

# pcd.estimate_normals()
# pcd.orient_normals_consistent_tangent_plane(100)
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# this has been the most crucial part, as


# create mesh1
# Visualize mesh


#write mesh to an STL file
o3d.io.write_triangle_mesh('SensorPcdDataGenerator.stl', mesh)


