import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from stl import mesh

x = np.linspace(-1,1,100)
print(x)
y = np.linspace(-1,1,100)
print(y)
z = np.linspace(0,0.5,100)
print(z)
x2d, y2d = np.meshgrid(x,y, indexing= 'xy')
x3d, y3d, z3d = np.meshgrid(x,y,z, indexing= 'xy')

mask = (x3d**2 + y3d**2 >= z3d) * (x3d**2 + y3d**2 <= 1.5*z3d)
mask[:,:,-1] = 0
mask[:,:,0] = 0
plt.figure(figsize=(5,5))
plt.pcolormesh(x2d, y2d, mask[:,:,20])
plt.show()

verts, faces, normals, values = measure.marching_cubes(mask,0)
obj_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))


for i, f in enumerate(faces):
    obj_3d.vectors[i] = verts[f]
obj_3d.save('3D_file.stl')
