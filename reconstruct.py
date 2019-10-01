from imutils import face_utils
import dlib
import cv2
import numpy as np
import pickle as pkl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

p = "/home/aman/Structure_From_Motion/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

path = input("Absolute Path of images : ")
import os
fnames = os.listdir(path)
W = np.zeros((len(fnames), 51, 2))

for index in range(1, len(fnames) + 1):
    image = cv2.imread(path +  str(index) + ".jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        W[index - 1] = shape[17:]
        for (x, y) in shape[17:]:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", image)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break


nFrames = len(fnames)
cv2.destroyAllWindows()

W_x = W[:,:,0]
W_y = W[:,:,1]
W = np.zeros((2*nFrames, 51))
W[:nFrames, :] = W_x
W[nFrames:2*nFrames, :] = W_y

nFeatures = 51

w_bar = W - np.mean(W, axis=1)[:, None]
w_bar = w_bar.astype('float32')

u, s_, v = np.linalg.svd(w_bar, full_matrices=False)
s = np.diag(s_)[:3, :3]
u = u[:, 0:3]
v = v[0:3, :]

S_cap = np.dot(np.sqrt(s), v)
R_cap = np.dot(u, np.sqrt(s))

number_of_frame = nFrames

R_cap_i = R_cap[0:number_of_frame, :]
R_cap_j = R_cap[number_of_frame:2 * number_of_frame, :]

A = np.zeros((2 * number_of_frame, 6))
i = 0
for i in range(number_of_frame):
    A[2 * i, 0] = (R_cap_i[i, 0] ** 2) - (R_cap_j[i, 0] ** 2)
    A[2 * i, 1] = 2 * ((R_cap_i[i, 0] * R_cap_i[i, 1]) - (R_cap_j[i, 0] * R_cap_j[i, 1]))
    A[2 * i, 2] = 2 * ((R_cap_i[i, 0] * R_cap_i[i, 2]) - (R_cap_j[i, 0] * R_cap_j[i, 2]))
    A[2 * i, 3] = (R_cap_i[i, 1] ** 2) - (R_cap_j[i, 1] ** 2)
    A[2 * i, 5] = (R_cap_i[i, 2] ** 2) - (R_cap_j[i, 2] ** 2)
    A[2 * i, 4] = 2 * ((R_cap_i[i, 2] * R_cap_i[i, 1]) - (R_cap_j[i, 2] * R_cap_j[i, 1]))

    A[2 * i + 1, 0] = R_cap_i[i, 0] * R_cap_j[i, 0]
    A[2 * i + 1, 1] = R_cap_i[i, 1] * R_cap_j[i, 0] + R_cap_i[i, 0] * R_cap_j[i, 1]
    A[2 * i + 1, 2] = R_cap_i[i, 2] * R_cap_j[i, 0] + R_cap_i[i, 0] * R_cap_j[i, 2]
    A[2 * i + 1, 3] = R_cap_i[i, 1] * R_cap_j[i, 1]
    A[2 * i + 1, 4] = R_cap_i[i, 2] * R_cap_j[i, 1] + R_cap_i[i, 1] * R_cap_j[i, 2]
    A[2 * i + 1, 5] = R_cap_i[i, 2] * R_cap_j[i, 2]
U, SIG, V = np.linalg.svd(A, full_matrices=False)
v = (V.T)[:, -1]

QQT = np.zeros((3, 3))

QQT[0, 0] = v[0]
QQT[1, 1] = v[3]
QQT[2, 2] = v[5]

QQT[0, 1] = v[1]
QQT[1, 0] = v[1]

QQT[0, 2] = v[2]
QQT[2, 0] = v[2]

QQT[2, 1] = v[4]
QQT[1, 2] = v[4]

Q = np.linalg.cholesky(QQT)

R = np.dot(R_cap, Q)

Q_inv = np.linalg.inv(Q)

S = np.dot(Q_inv, S_cap)

X = S[0, :]
Y = S[1, :]
Z = S[2, :]


pointcloud = np.zeros((X.shape[0], 3))
pointcloud[:,0] = X
pointcloud[:,1] = Y
pointcloud[:,2] = Z

import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pointcloud)
o3d.io.write_point_cloud("pointcloud.ply", pcd)
pcd_load = o3d.io.read_point_cloud("pointcloud.ply")
o3d.visualization.draw_geometries([pcd_load])


