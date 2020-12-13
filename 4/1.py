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
    # F = [e2]_x H
    H = np.random.uniform(0, 5, (3, 3))
    e2 = np.append(np.random.uniform(0, 5, (2, 1)), [[1]], axis=0)
    F = np.cross(e2, H.T).T
    print("H = \n", H)
    print("e2 = \n", e2)
    print("F = \n", F)

def task_two():
    """Epipole on second image have
    e2 = (1, 1, 1).T coords. Find homography transform
    which move e2 to (1, 0, 0).T"""
    # If e2 is x shift
    # e2 = (f, 0, 1).T thus
    #     | 1   0 0 |
    # G = | 0   1 0 |
    #     | -1/f 0 1 |
    # and (1, 0, 0).T = G e2
    # But first we need to move our point to
    # make y coord equal 0
    # e2 = (1, 1, 1).T         | 1 0  0|
    # (1, 0, 0).T = G Ty = G * | 0 1 -1| * e2 =
    #                          | 0 0  1|
    #   |  1  0  0 |
    # = |  0  1 -1 | e2
    #   | -1  0  1 |
    F = np.array([[1, 0, 0],
                  [0, 1, -1],
                  [-1, 0, 1]])
    print("F = \n", F)

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
    """There is two cameras
    shifted to each other by (f, 0, 0).T and
    both cameras has identity inside params matrix.
    Find fundamental matrix by 8 given
    projection accord points"""
    # Build cameras
    E = np.diag(np.ones((3, )))
    T1 = np.array([[0, 0, 0]]).T
    T2 = np.array([[np.random.uniform(1, 10), 0, 0]]).T
    print("X shift: \n", T2)
    P1 = np.append(E, T1, axis=1)
    P2 = np.append(E, T2, axis=1)
    # Eight random 3d points
    Q = np.random.uniform(3, 10, (8, 3, 1))
    print("Q = \n", Q)
    Q1 = np.array([P1.dot(np.append(x, [1])) for x in Q])
    Q2 = np.array([P2.dot(np.append(x, [1])) for x in Q])
    # Build equation for fundamental matrix
    W = list()
    for q1, q2 in zip(Q1, Q2):
        q1 /= q1[2]
        q2 /= q2[2]
        W += [[q1[0] * q2[0],
               q1[1] * q2[0],
               q2[1],
               q1[0] * q2[1],
               q1[1] * q2[1],
               q2[1],
               q1[0],
               q1[1],
               1
               ]]
    W = np.array(W)
    # Get a pseudo reverce matrix
    w, u, vt = cv.SVDecomp(W)
    # Av_i = sigma_i * u_i
    # hence
    # all solutions for A x = 0  is
    # v_i, corresponds to zero sigma value
    # Find count of zero sigma
    sol_dim = len(np.argwhere(w < 1e-5))
    # Get vertices
    sol = vt[-sol_dim:]

    Fp = np.array(sol[0]).reshape(3, 3)
    # Find closest orthogonal matrix
    w, u, vt = cv.SVDecomp(Fp)
    F = np.dot(u.dot(np.diag(np.append(w.flatten()[:2], [0]))), vt)
    print("F = \n", F)

if __name__ == '__main__':
    tasks = [task_one, task_two, task_three, task_four]
    for i, task in enumerate(tasks):
        print("-" * 20 + " Task ", i + 1, "-" * 20)
        task()