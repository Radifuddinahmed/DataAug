import numpy as np
# from SensorReadData import SensRead
import open3d as o3d
# from sample_ImageAcquisition_4_pcdFileGenerator import main
import matplotlib.pyplot as plt

# list_of_arrays = []
#
# y_val = 0
# z = 0
# while(z<2):
#
#     #read the data from the sensor read file
#     # x_val, z_val = SensRead()
#     x_val, z_val = main()
#
#
#
#     points = [None, None, None]
#     vertices = np.array(points)
#
#
#
#     for i in range(0,3):
#         vertices[0] = (x_val[i])
#         vertices[1] = (y_val)
#         vertices[2] = (z_val[i])
#
#         array = list(vertices)
#         # Append the array to the list_of_arrays
#         list_of_arrays.append(array)
#     # print(points)
#     print(list_of_arrays)
#     y_val = y_val + 1
#     z = z+1
#
#
# # #write vertices as pcd file
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(list_of_arrays)
# o3d.io.write_point_cloud("SensorPcdDataGenerator.pcd", pcd)
# o3d.visualization.draw_geometries([pcd])
#
#
#
#
import serial
import time
timeDelayComm = 2

def stepForward():
    ser = serial.Serial('COM3', 9600)
    time.sleep(timeDelayComm)
    ser.write("f".encode())
    ser.write("s".encode())
    ser.flush()
    ser.close()
def stepBackward():
    ser = serial.Serial('COM3', 9600)
    time.sleep(timeDelayComm)
    ser.write("b".encode())
    ser.write("s".encode())
    ser.flush()
    ser.close()

stepForward()
stepForward()
stepForward()
stepForward()

stepBackward()



# list = []
#
#
# for i in range(0,10):
#     for z in range(1000,1003):
#         list.insert(i,z)
#         for y in range(100, 103):
#             list.insert(i,y)
#             for x in range(0, 3):
#                 list.insert(i,x)
#
#
# print(list)