# 3D Computer Vision Course HSE Mag Data Mining 20
# Third exercise
# Daniil Lyakhov
# dupeljan@gmail.com
# dalyakhov@edu.hse.ru

import numpy as np
import cv2 as cv

def task_one(silent= False):
    """Find fundamental matrix if
    first cam have projection matrix
    P = [E|0]
    and second is rotate by 45 deg around OZ,
    translated on 10 by OX and have E as inside
    cam parameters, have projection matrix
    P = [R|T]
    Remember that K and K` - inside cam
    param matrix is identity matrix
    """
    # The formula is following:
    # F = K`^(-T) [T]_x R K^(-1)
    # but since K = K` = E
    # F = [T]_x R
    tetta = 45 * np.pi / 180
    R = np.array([[np.cos(tetta), -np.sin(tetta), 0],
                  [np.sin(tetta),  np.cos(tetta), 0],
                  [0, 0, 1]])
    T = np.array([[10, 0, 0]]).T
    # F = [T]_x R
    #                      VVVVChoose columnVVVVV
    F = np.array([np.cross(T.flatten(), R[:, i]) for i in range(3)])
    if not silent:
        print("F =\n", F)
        return
    return {"F": F, "R": R, "T": T}


def task_two():
    """Find fundamental matrix between
    two cams with some params"""
    tetta1 = 45 * np.pi / 180
    R1 = np.array([[np.cos(tetta1), -np.sin(tetta1), 0],
                  [np.sin(tetta1), np.cos(tetta1), 0],
                  [0, 0, 1]])
    T1 = np.array([[0, 0, 0]]).T
    tetta2 = -tetta1
    R2 = np.array([[np.cos(tetta1), 0, np.sin(tetta1)],
                   [0, 1, 0],
                   [-np.sin(tetta1), 0, np.cos(tetta1)]])
    T2 = np.array([[0, 10, 0]]).T
    # Formula to calculate:
    # F = [e2]_x P2 P1_rev
    # P1, P2 - first and second projection matrix
    P1 = np.dot(np.diag(np.ones((3, ))), np.append(R1, T1, axis=1))
    P2 = np.dot(np.diag(np.ones((3, ))), np.append(R2, T2, axis=1))

    # Find e2 = P2 * O1
    O1 = np.array([[0, 0, 0]]).T + T1
    e2 = P2.dot(np.append(O1, np.array([1])))

    # Find revers P1
    w, u, vt = cv.SVDecomp(P1)
    # u w vt r  = E
    # r = vt.T w^-2 u.T
    P1_rev = np.dot(vt.T, np.dot(np.diag([1 / x if x > 1e-5 else 0 for x in w.flatten()]), u.T))

    # Calc F
    P = P2.dot(P1_rev)
    F = np.array([np.cross(e2, P[:, i]) for i in range(3)])
    F = np.cross(e2, P)
    print("F = \n", F)


def task_three():
    """Find all epiphole in task_one"""
    tetta = 45 * np.pi / 180
    R = np.array([[np.cos(tetta), -np.sin(tetta), 0],
                  [np.sin(tetta), np.cos(tetta), 0],
                  [0, 0, 1]])
    T = np.array([[10, 0, 0]]).T
    # Find coords of center of
    # the second camera
    O2 = np.array([[0, 0, 0]]).T + T

    #O2 = np.append(O2, np.array([1]))
    # e = [I|0] O2 = O2
    print("e1 = \n", O2)
    # e_hatch = K2 T = T
    print("e2 = \n", T)

def task_four():
    """Find ephypolar line through
    point (0, 0) on first camera image and correspond
    line on the second camera image from task 1"""
    # L1 = q2T F
    # L2 = F q1
    # F already found in task three
    # q1 is (0, 0, 1)
    # q2 = P2 P1_rev q
    t1 = task_one(silent=True)
    q1 = np.array([[0, 0, 1]]).T
    P1_rev = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [0, 0, 0]])
    P2 = np.append(t1["R"], t1["T"], axis=1)
    q2 = np.dot(P2.dot(P1_rev), q1)
    L1 = q2.T.dot(t1["F"]).T
    L2 = t1["F"].dot(q1)
    print("L1 = \n", L1)
    print("L2 = \n", L2)

def task_five():
    """Find L2 with assumption that
    camera 2 is translated by ox (
    thus F = | 0, 0, 0 |
             | 0, 0,-1 |
             | 0, 1, 0 |
    ) and L1 = (0, 1, 0).T"""
    # L1 = q2.T F
    # thus q2[2] = 1
    #      q2[1] = 0
    #      q2[0] = 0
    #
    # q2.T F q1 = 0
    # thus [[0, 1, 0]].dot(q1) = 0
    # thus q1 = [[a, 0, 1]].T
    #
    # L2 = Fq
    # thus L2 = [[0, -1, 0]].T
    print("L2 = \n",
          np.array([[0, -1, 0]]).T)

if __name__ == '__main__':
    tasks = [task_one, task_two, task_three, task_four, task_five]
    for i, task in enumerate(tasks):
        print("-" * 20 + " Task ", i + 1, "-" * 20)
        task()

