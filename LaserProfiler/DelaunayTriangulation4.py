import numpy as np
import open3d as o3d
def reconstruct_surface_from_pcd(pcd_file_path):
    # Load the point cloud data from the given file path.
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    # Downsample the point cloud to make computations faster, using voxels of size 0.05.
    down_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # Estimate the normals of the downsampled point cloud.
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))

    # Orient the normals towards a specific camera location.
    camera_location = np.array([0, 0, -1])
    down_pcd.orient_normals_towards_camera_location(camera_location)

    # Visualize the point cloud and its normals (optional).
    o3d.visualization.draw_geometries([down_pcd], point_show_normal=True)

    # Use Poisson surface reconstruction to create a mesh from the point cloud.
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(down_pcd, depth = 12)

    # Process densities (optional, for removing low-density vertices).
    densities = np.asarray(densities)
    density_threshold = 0.0095
    vertices_to_remove = densities < np.quantile(densities, density_threshold)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Visualize the reconstructed surface
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # The following lines save the reconstructed mesh to a file and print the file path.
    # mesh_file_path = pcd_file_path.replace('.pcd', '_reconstructed_mesh.obj')
    # o3d.io.write_triangle_mesh(mesh_file_path, mesh)
    # print(f'Reconstructed mesh saved to: {mesh_file_path}')


# Input
pcd_file_path = "SensorPcdDataGenerator20240317181729.pcd"
reconstruct_surface_from_pcd(pcd_file_path)