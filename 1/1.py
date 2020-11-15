# 3D Computer Vision Course HSE Mag Data Mining 20
# Second exercise
# Daniil Lyakhov
# dupeljan@gmail.com
# dalyakhov@edu.hse.ru

import numpy as np
import cv2 as cv

def task_one():
    """Find camera projection matrix by
    all params given"""
    # Rotation about OZ by 45 degree
    tetta = 45 * np.pi / 180
    R = np.array([[np.cos(tetta), -np.sin(tetta), 0],
                  [np.sin(tetta),  np.cos(tetta), 0],
                  [0, 0, 1]])
    T = np.array([[0, 0, 10]]).T
    K = np.array([[400,   0, 960],
                  [  0, 400, 540],
                  [  0,   0,   1]])
    P = np.dot(K, np.append(R, T, axis=1))
    print("Projection matrix P:\n", P)
    # Point to test
    point = np.array([[10, -10, 100]]).T
    print("Point coords in global scope:\n", point)
    point = np.dot(P, np.append(point, np.ones((1, 1)), axis=0))
    # Normalize
    point = np.array([[np.round(p/point[2][0])] for p in point[:-1, 0]])
    print("Point in the camera scope:\n", point)

def task_two(points):
    """Find homography matrix from set of points pairs
    by direct linear transformation and SVD
    params:
        points: set of points"""
    assert len(points) == 4, "Given too much point to calculate"
    assert isinstance(points[0], tuple) and len(points[0][0]) == len(points[0][1]) == 2,\
                                        "Inappropriate stucture of points set"
    # P` = H * P where
    #   P - initial points
    #   P` - homography translated points
    # P` = H * U * S * VT
    # H ~ P` * V * S^(-1) * UT
    P = np.array([[x[0, 0], x[0, 1], 1] for x in points]).T
    P_tilda = np.array([[x[1, 0], x[1, 1], 1] for x in points]).T
    W, U, VT = cv.SVDecomp(P)
    W_rev = np.diag(1 / W.flatten())
    H = P_tilda.dot(VT.T).dot(W_rev).dot(U.T)
    print("Homography matrix:\n", H)


def task_three():
    """Find homography transform matrix
    between two pictures from one camera with
    30 degree rotation around OX from camera origin"""
    # Formula to calculate:
    # q2 = (z2 / z1) * (R + T * nt / d) * q1
    # where R - rotation
    #       T - translation
    #       nt - normal vertex of common plane of the 3d points
    #       d  - shift of the common plane
    #       and (R + T * nt / d) required homography transform
    #                            defined up to constant
    # But in our case T == 0
    tetta = 30 * np.pi / 180
    H = np.array([[1, 0, 0]
                  [0, np.cos(tetta), -np.sin(tetta)],
                  [0, np.sin(tetta), np.cos(tetta)],
                  ])
    print("Homography transformation:\n", H)

if __name__ == '__main__':
    task_one()