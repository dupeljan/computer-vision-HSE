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
    F = np.cross(e2, H, axisa=0, axisb=0, axisc=0)
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
    T2 = np.array([[10, 0, 0]]).T
    print("X shift: \n", T2)
    P1 = np.append(E, T1, axis=1)
    P2 = np.append(E, T2, axis=1)
    # Eight random 3d points
    num_points = 8
    Q = np.append(np.array([np.random.rand(3) * 10 for j in range(num_points)]),
                  np.ones(num_points).reshape(num_points, 1), axis=1)
    q1 = np.matmul(P1, Q.T)
    q2 = np.matmul(P2, Q.T)

    # normalize in
    Q1 = (q1 / q1[2]).T
    Q2 = (q2 / q2[2]).T
    equations = []
    for pts1, pts2 in zip(Q1, Q2):
        u1 = pts1[0]
        v1 = pts1[1]
        u2 = pts2[0]
        v2 = pts2[1]
        equations.append([u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1, 1])

    equations = np.array(equations).T
    U,E,V = np.linalg.svd(equations)

    # Av_i = sigma_i * u_i
    # hence
    # all solutions for A x = 0  is close to
    # v_i, corresponds to zero sigma value
    f = U[:, -1]

    Fp = np.array(f).reshape(3, 3)
    # Find closest orthogonal matrix
    w, u, vt = cv.SVDecomp(Fp)
    F = np.dot(u.dot(np.diag(np.append(w.flatten()[:2], [0]))), vt)
    print("F = \n", F)


if __name__ == '__main__':
    tasks = [task_one, task_two, task_three, task_four]
    for i, task in enumerate(tasks):
        print("-" * 20 + " Task ", i + 1, "-" * 20)
        task()