# 3D Computer Vision Course HSE Mag Data Mining 20
# First exercise
# Daniil Lyakhov
# dupeljan@gmail.com
# dalyakhov@edu.hse.ru

import numpy as np
import cv2 as cv


def task_one():
    """Find closes orthogonal matrix
    in Frobenious distance space by SVD """
    A = np.array([[0.5, 2.16506351, 0.4330127],
                  [-0.8660254, 1.25, 0.25],
                  [0, 0.5, 2.5]])
    w, u, vt = cv.SVDecomp(A)
    c = np.dot(u, vt)
    print("A = \n", A)
    print("Closest orthogonal matrix c:\n ", c)
    print("Orthogonality proof: np.dot(c, c.T) \n", np.dot(c, c.T))
    print("This matrix rotate vectors by an angle θ about the oz with angle ",
          180 * np.arcsin(c[1, 0]) / np.pi, " degree")


def task_two():
    """ Find revers matrix for two given by SCD"""
    for n in [3, 10]:
        A = 1 / np.array([np.arange(n) + x + 1 for x in range(n)])
        w, u, vt = cv.SVDecomp(A)
        w_rev = np.diag(1 / w.flatten())
        # u w vt r  = E
        # r = vt.T w^-1 u.T
        r = np.dot(vt.T, np.dot(w_rev, u.T))
        print("A = \n", A)
        print("Reverse A matrix r: \n", r)
        print("np.dot(A, r): \n", np.dot(A, r))


def task_three():
    """ Find all solutions  Ax = 0"""
    n = 4
    A = np.array([np.arange(n) + x + 1 for x in range(n)], dtype=np.float64)
    print("A = \n", A)
    w, u, vt = cv.SVDecomp(A)
    # Av_i = sigma_i * u_i
    # hence
    # all solutions for A x = 0  is
    # v_i, corresponds to zero sigma value
    # Find count of zero sigma
    sol_dim = len(np.argwhere(w < 1e-5))
    # Get vertices
    sol = vt[-sol_dim:]
    print("Solutions:")
    for s in sol:
        print("s = const *", s)
        print("----check----\nnp.dot(A,s): \n", np.dot(A, s.T), "\n-------------")


def task_four():
    """ Find two lines intersection points """
    l1, l2 = np.random.rand(3), np.random.rand(3)
    # Perform vector mul
    c = np.cross(l1, l2)
    # Translate solution into plane
    sol = c[0]/c[2], c[1]/c[2], 1
    print("First line: {}x + {}y + {}".format(*l1))
    print("Second line: {}x + {}y + {}".format(*l2))
    print("Interception point: ", sol)
    print("----Check----")
    res = np.dot(sol, l1), np.dot(sol, l2)
    print("Line 1 and point scalar product: ", res[0])
    print("Line 2 and point scalar product: ", res[1])
    print("If line dot point is equal to zero then this point on the line")


if __name__ == '__main__':
    task_one()
    task_two()
    task_three()
    task_four()
