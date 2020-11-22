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

def task_two_test():
    """Gen data to test task 3"""
    # First test
    # Create points list for task two
    points = np.random.rand(2, 4)
    # Translate and rotate it somehow
    tetta = np.random.uniform(low=0, high=2 * np.pi, size=(1,))[0]
    R = np.array([[np.cos(tetta), -np.sin(tetta)],
                  [np.sin(tetta), np.cos(tetta)]])
    T = np.random.uniform(low=0, high=3, size=(2, 1))
    H = np.append(R, T, axis=1)
    points_translated = np.dot(H, np.append(points, np.ones((1, 4)), axis=0))
    print("Points 2d translation + rotation:\n", H)
    points_list = np.array(list(zip(points.T, points_translated.T)))
    task_two(points_list)
    # Second test
    H = np.random.rand(3, 3)
    points_translated = np.dot(H, np.append(points, np.ones((1, 4)), axis=0))
    # Normalize it
    points = np.random.rand(3, 4)
    tetta = np.random.uniform(low=0, high=2 * np.pi, size=(1,))[0]
    R = np.array([[np.cos(tetta), -np.sin(tetta), 0],
                  [np.sin(tetta), np.cos(tetta), 0],
                  [0, 0, 1]])
    T = np.random.uniform(low=0, high=3, size=(3, 1))
    H = np.append(R, T, axis=1)
    print("Points 3d translation + rotation:\n", H)
    points_translated = np.dot(H,  np.append(points, np.ones((1, 4)), axis=0))
    # Convert to p2
    norm = lambda x: [x[0] / x[2], x[1] / x[2]]
    points = np.array([norm(x) for x in points.T]).T
    points_translated = np.array([norm(x) for x in points_translated.T]).T
    points_list = np.array(list(zip(points.T, points_translated.T)))
    task_two(points_list)

def task_two(points):
    """Find homography matrix from set of points pairs
    by direct linear transformation and SVD
    params:
        points: array of points"""
    assert len(points) == 4, "Given too much point to calculate"
    assert len(points[0][0]) == len(points[0][1]) == 2,\
                                        "Inappropriate stucture of points list"
    # P` = H * P where
    #   P - initial points
    #   P` - homography translated points
    # P` = H * U * S * VT
    # H ~ P` * V * S^(-2) * UT
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
    H = np.array([[1, 0, 0],
                  [0, np.cos(tetta), -np.sin(tetta)],
                  [0, np.sin(tetta), np.cos(tetta)],
                  ])
    print("Homography transformation:\n", H)


if __name__ == '__main__':
    tasks = [task_one, task_two_test, task_three]
    for i, task in enumerate(tasks):
        print("-"*20 + " Task ", i+1, "-"*20)
        task()
