# ******************************************************************************
# *                                                                            *
# *                 Depth-Map-Based Surface Reconstruction                     *
# *                             Iftesam Nabi                                   *
# *                                                                            *
# *  Description:                                                              *
# *  This script transforms a cloud of points from a 3D scan into a smooth     *
# *  surface, mimicking the original shape. First, it spreads the points       *
# *  across a grid, creating a map showing the height of each point. Next,     *
# *  it connects the dots, forming a mesh outlining the shape of the surface.  *
# *  If the result looks rough, it smooths it out, making it more like the     *
# *  original surface. Finally, it visualizes this smoothed shape on the       *
# *  screen. From scattered points, it creates a neat, visual model of the     *
# *  object.                                                                   *
# *                                                                            *
# ******************************************************************************
# *                                                                            *
# *  1. point_cloud_to_depth_map(pcd, resolution):                             *
# *     Converts a point cloud into a depth map. The function iterates through *
# *     the point cloud, computing a grid-based depth map that captures the    *
# *     highest point (z-value) within each grid cell. This transformation     *
# *     enables a structured representation of the surface, facilitating the   *
# *     subsequent reconstruction process.                                     *
# *                                                                            *
# *  2. reconstruct_surface_from_depth_map(depth_map, resolution):             *
# *     Reconstructs a surface mesh from the depth map. For each cell in the   *
# *     depth map with a valid depth value, vertices are generated at the cell *
# *     location, with z-values corresponding to the depth. Faces (triangles)  *
# *     are then created between adjacent vertices, forming a mesh that        *
# *     approximates the surface represented by the depth map.                 *
# *                                                                            *
# *  3. laplacian_smoothing(verts, faces, alpha, iterations):                  *
# *     Applies Laplacian smoothing to the reconstructed mesh to enhance its   *
# *     visual quality. By adjusting each vertex towards the average position  *
# *     of its neighbors, this smoothing process reduces mesh roughness,       *
# *     yielding a smoother surface.                                           *
# *                                                                            *
# *  4. load_pcd_and_generate_depth_map(pcd_file_path, resolution):            *
# *     Loads a point cloud from a specified file path and optionally          *
# *     downsamples it to simplify the data. It then calls                     *
# *     point_cloud_to_depth_map to generate a depth map and point map for     *
# *     further processing.                                                    *
# *                                                                            *
# *  5. visualize_mesh(verts, faces):                                          *
# *     Visualizes the reconstructed (and optionally smoothed) mesh using      *
# *     Open3D. The function creates a TriangleMesh object, assigns it         *
# *     vertices and faces, computes vertex normals for better lighting        *
# *     effects, and displays the mesh in a visualization window named         *
# *     "Depth Based Surface Reconstruction".                                  *
# *                                                                            *
# *  The global_resolution variable sets the resolution for the depth map and  *
# *  reconstruction process, balancing detail against performance.By adjusting *
# *  this value, users can tailor the algorithm's output to meet specific      *
# *  requirements for surface detail and computational efficiency.             *
# *                                                                            *
# ******************************************************************************

import numpy as np
import open3d as o3d

global_resolution = 1
def point_cloud_to_depth_map(pcd, resolution=global_resolution):
    # Convert point cloud data to a NumPy array for processing.
    points = np.asarray(pcd.points)

    # Determine the minimum and maximum coordinates of the point cloud for boundary calculations.
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # Calculate the necessary size of the depth map based on point cloud bounds and desired resolution.
    size_x = int(np.ceil((max_bound[0] - min_bound[0]) / resolution))
    size_y = int(np.ceil((max_bound[1] - min_bound[1]) / resolution))

    # Initialize the depth map with NaN values and a point map to store point indices.
    depth_map = np.full((size_x, size_y), np.nan)
    point_map = [[[] for _ in range(size_y)] for _ in range(size_x)]

    # Populate the depth map and point map with the highest point (z-value) in each grid cell.
    for i, point in enumerate(points):
        x_idx = int((point[0] - min_bound[0]) / resolution)
        y_idx = int((point[1] - min_bound[1]) / resolution)

        # Update the depth value if the current point is higher than the existing value.
        depth_map[x_idx, y_idx] = max(depth_map[x_idx, y_idx], point[2]) if not np.isnan(depth_map[x_idx, y_idx]) else point[2]
        point_map[x_idx][y_idx].append(i)

    return depth_map, point_map

def reconstruct_surface_from_depth_map(depth_map, resolution=global_resolution):
    # Determine the number of rows and columns in the depth map.
    rows, cols = depth_map.shape
    verts = []  # List to store vertex coordinates.
    faces = []  # List to store faces (triangles).

    # Generate vertices from the depth map values.
    for i in range(rows):
        for j in range(cols):
            # Only process cells with valid depth values.
            if not np.isnan(depth_map[i, j]):
                # Calculate world coordinates from depth map indices.
                x = i * resolution
                y = j * resolution
                z = depth_map[i, j]
                verts.append([x, y, z])

    # Convert the list of vertices to a NumPy array for efficiency.
    verts = np.array(verts)

    # Function to compute a linear index for a vertex in the grid.
    def vertex_index(i, j):
        return i * cols + j

    # Generate faces by creating two triangles for each cell that forms a quad with its right and bottom neighbors.
    for i in range(rows - 1):
        for j in range(cols - 1):
            # Ensure all corners of the quad have valid depth values before creating triangles.
            if not np.isnan(depth_map[i, j]) and not np.isnan(depth_map[i + 1, j]) and not np.isnan(depth_map[i, j + 1]) and not np.isnan(depth_map[i + 1, j + 1]):
                # Calculate vertex indices for the corners of the quad.
                v1 = vertex_index(i, j)
                v2 = vertex_index(i + 1, j)
                v3 = vertex_index(i, j + 1)
                v4 = vertex_index(i + 1, j + 1)
                # Add two faces (triangles) to cover the quad area.
                faces.append([v1, v2, v3])
                faces.append([v2, v4, v3])

    # Convert the list of faces to a NumPy array.
    faces = np.array(faces)

    return verts, faces

def laplacian_smoothing(verts, faces, alpha, iterations):
    # Initialize a dictionary to store the set of neighboring vertex indices for each vertex.
    vert_neighbors = {i: set() for i in range(len(verts))}

    # Populate the neighbors dictionary. For each face, add each vertex as a neighbor of the others.
    for face in faces:
        for i in face:
            for j in face:
                if i != j:
                    vert_neighbors[i].add(j)

    # Perform the smoothing operation for the specified number of iterations.
    for _ in range(iterations):
        # Create a copy of the current vertices to hold the updated positions.
        new_verts = np.copy(verts)
        # Update each vertex position based on the average of its neighbors.
        for i, neighbors in vert_neighbors.items():
            if len(neighbors) == 0: continue  # Skip isolated vertices with no neighbors.
            # Calculate the average position of the neighbors.
            neighbor_verts = np.mean(verts[list(neighbors)], axis=0)
            # Move the vertex towards the average position of its neighbors, controlled by alpha.
            new_verts[i] = verts[i] + alpha * (neighbor_verts - verts[i])
        # Update the vertices with their new positions after this iteration.
        verts = new_verts

    # Return the smoothed vertices.
    return verts

def load_pcd_and_generate_depth_map(pcd_file_path, resolution= global_resolution):
    # Load the point cloud from the specified file path using Open3D.
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    #  downsample the point cloud to reduce complexity and speed up processing for visualization.
    down_pcd = pcd.voxel_down_sample(voxel_size=1)

    # Generate a depth map and a point map from the potentially downsampled point cloud.
    depth_map, point_map = point_cloud_to_depth_map(down_pcd, resolution=resolution)

    # Return the generated depth map and point map for further processing.
    return depth_map, point_map

def visualize_mesh(verts, faces):
    # Create a new TriangleMesh object from Open3D.
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Initialize a Visualizer object.
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Depth-Map-Based Surface Reconstruction")
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def export_mesh_to_stl(verts, faces, file_path="output_mesh.stl"):
    """
    Exports the mesh defined by vertices and faces to an STL file.

    Parameters:
    - verts: np.ndarray. The vertices of the mesh.
    - faces: np.ndarray. The faces of the mesh.
    - file_path: str. The path to the output STL file.
    """
    # Create the mesh object with the given vertices and faces
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()  # Optional: Computes vertex normals for better shading

    # Export the mesh to an STL file in binary format
    o3d.io.write_triangle_mesh(file_path, mesh)
    print(f"Mesh exported to STL file at: {file_path}")


# Input file path for the point cloud data.
pcd_file_path = "SensorPcdDataGenerator20240317181729.pcd"

# Generate a depth map from the point cloud.
depth_map, point_map = load_pcd_and_generate_depth_map(pcd_file_path, resolution=global_resolution)

# Reconstruct the surface from the depth map into vertices and faces.
verts, faces = reconstruct_surface_from_depth_map(depth_map, resolution=global_resolution)

# refine the mesh using Laplacian smoothing to create a smoother appearance.
smoothed_verts = laplacian_smoothing(verts, faces, alpha=0.0, iterations=1)

# Visualize the original or smoothed mesh.
visualize_mesh(smoothed_verts, faces)

export_mesh_to_stl(smoothed_verts, faces, "./STL_Renders/Depth_Based_STL_output.stl")