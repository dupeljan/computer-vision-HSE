# 3D Computer Vision Course HSE Mag Data Mining 20
# Third exercise
# Daniil Lyakhov
# dupeljan@gmail.com
# dalyakhov@edu.hse.ru

import numpy as np
import cv2 as cv
EPS = 1e-5

def task_one():
    """Given homographic matrix between
    two points and epipolar coords of second image.
    Calculate fundamental matrix"""
    H = np.random.uniform(0, 5, (3, 3))
    e2 = np.append(np.random.uniform(0, 5, (2, 1)), [[1]], axis=0)
    F = np.cross(e2.T, H)
    print("H = \n", H)
    print("e2 = \n", e2)
    print("F = \n", F)

def task_two():
    """Epipole on second image have
    e2 = (1, 1, 1).T coords. Find homography transform
    which move e2 to (1, 0, 0).T"""
    # e1 = H e2
    # H * (1, 1, 1).T = (1, 0, 0).T
    pass

def task_three():
    """Second camera shifted by x related to
    first camera on T.
    Given p1 and p2 - camera coords of global
     point Q. Camera params F1 and F2 are different"""
    # Z = fx * Tx / (q2[0] - q1[0])
    min_, max_ = 0, 10
    T = np.array([np.random.uniform(min_, max_), 0, 0]).T
    fx = np.random.uniform(EPS, max_)
    q1 = np.random.uniform(min_, max_, (3, 1))
    q1[2] = 1 if abs(q1[2]) < EPS else q1[2]
    q1 /= q1[2]
    q2 = np.random.uniform(min_, max_, (3, 1))
    q2[2] = 1 if abs(q2[2]) < EPS else q2[2]
    q2 /= q2[2]
    if q1[0] > q2[0]:
        q1, q2 = q2, q1
    print("T = ", T)
    print("fx = ", fx)
    print("q1 = \n", q1)
    print("q2 = \n", q2)
    z = fx * T[0] / (q2[0] - q1[0])
    print("z = ", z)

def task_four():
    pass

if __name__ == '__main__':
    tasks = [task_one, task_two, task_three, task_four]
    for i, task in enumerate(tasks):
        print("-" * 20 + " Task ", i + 1, "-" * 20)
        task()