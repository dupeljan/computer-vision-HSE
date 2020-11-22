# [a]_x_(i, k) = sum_j eps_(i, j, k) a_j =
    #   |   0  -a3   a2 |
    # = |  a3    0  -a1 |
    #   | -a2   a1    0 |
"""    T_x = np.array([[0, -T[2], T[1]],
                    [T[2], 0, -T[0]],
                    [-T[1], T[0], 0]])
    F = T_x.dot(R)"""