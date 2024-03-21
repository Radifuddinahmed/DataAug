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


######### alpha shapes
# Read and visualize point cloud data
pcd = o3d.io.read_point_cloud("SensorPcdDataGenerator20240317181729.pcd")
o3d.visualization.draw_geometries([pcd])
pcd.estimate_normals()
pcd.orient_normals_consistent_tangent_plane(100)
# the lines above created a tangent based normal, before using this the normals were not tangent
# hence the surface of the sphere were not filled
o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# ########## mesh creation and visualization using the "alpha shapes method"
# alpha = 0.2
#     #0.03 has holes
#     #0.09 has holes
#     #0.1 has holes
#     #0.5 shows best results with full surface
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


# ######## mesh creation and visualization using the "ball pivoting algorithm"
# radii = [0.02]
# #0.005!not filled, 0.01!partially filled, 0.02!partially filled, 0.04!partially filled
# #.2 .3 .4
# # previous values did not work due to surfaces being partially normal
# ## now with the new tangent based normal algorithm it is working now
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#     pcd, o3d.utility.DoubleVector(radii))
# o3d.visualization.draw_geometries([pcd, mesh])



######## mesh creation and visualization using the "Poisson Surface Reconstruction"
o3d.visualization.draw_geometries([pcd],
                                  zoom=1.4, # zoom value
                                  front=[-0.4761, -0.1698, -0.4434],
                                  lookat=[1.05, 0.8, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])

print('run Poisson surface reconstruction')
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9) # tried values from .1 to 14, nothing worked
print(mesh)
o3d.visualization.draw_geometries([mesh],
                                  zoom=1.4, # zoom value
                                  front=[-0.4761, -0.1698, -0.4434],
                                  lookat=[1.05, 0.8, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])

print('visualize densities')
densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
o3d.visualization.draw_geometries([density_mesh],
                                  zoom=1.4, # zoom value
                                  front=[-0.4761, -0.1698, -0.4434],
                                  lookat=[1.05, 0.8, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])

print('remove low density vertices')
vertices_to_remove = densities < np.quantile(densities, 0.01)
mesh.remove_vertices_by_mask(vertices_to_remove)
print(mesh)
o3d.visualization.draw_geometries([mesh],
                                  zoom=1.4, # zoom value
                                  front=[-0.4761, -0.1698, -0.4434],
                                  lookat=[1.05, 0.8, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])

########## Normal Estimation

# pcd.estimate_normals()
# pcd.orient_normals_consistent_tangent_plane(100)
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)

# this has been the most crucial part, as


# create mesh1
# Visualize mesh


# #write mesh to an STL file
# o3d.io.write_triangle_mesh('SensorPcdDataGenerator.stl', mesh)


