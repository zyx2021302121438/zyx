import numpy as np
import math
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from mpl_toolkits.mplot3d import Axes3D


def xs(x, y, w):
    if len(x) == len(y) == len(w):
        num1 = 0
        num2 = 0
        for k in range(len(x)):
            num1 += x[k] * y[k] * w[k]
            num2 += abs(x[k] * y[k] * w[k])
        if num2 == 0:
            return 0
        else:
            return num1 / num2
    else:
        print('error')


def Target_node_weight(edges):
    dic = {}
    for i in edges:
        dic.setdefault(int(i.split('\t')[0]), [])
        dic[int(i.split('\t')[0])].append(int(i.split('\t')[2][:-1]))
    dic0 = sorted(dic.items(), key=lambda item: item[0])
    n1 = len(dic0)
    n2 = len(dic0[0][1])
    w1 = []
    w2 = []
    for i in range(n2):
        x = 0
        y = 0
        for j in dic0:
            x += abs(j[1][i])
            if j[1][i] > 0:
                y += j[1][i]
        w1.append(abs(0.5 - y / x))

    for k in range(n2):
        ph = 0
        cs = -1
        for v in range(n2):
            f1 = 0
            f2 = 0
            f3 = 0
            f4 = 0
            f5 = 0
            for i in range(len(dic0) - 1):
                for j in range(i + 1, len(dic0)):
                    die = [dic0[i][1][k], dic0[j][1][k], dic0[j][1][v], dic0[i][1][v]]
                    if die == [1, 1, 1, 1]:
                        f1 += 1
                    elif die == [-1, -1, -1, -1]:
                        f2 += 1
                    elif die.count(-1) == 2 and die.count(1) == 2:
                        f3 += 1
                    elif die.count(-1) == 1 and die.count(1) == 3:
                        f4 += 1
                    elif die.count(-1) == 3 and die.count(1) == 1:
                        f5 += 1
            fs = f1 + f2 + f3 + f4 + f5
            if (f4 + f5) / fs < 0.3:
                ph += (f4 + f5) / fs
                cs += 1
        if cs == 0:
            w2.append(0.3)
        else:
            w2.append(ph / cs)

    for i in range(n2):
        if w1[i] < 0.25 and w2[i] < 0.15:
            w1[i] = 14
        elif w1[i] < 0.25 and 0.22 < w2[i] < 0.29:
            w1[i] = 14
        elif w1[i] < 0.25 and 0.15 < w2[i] < 0.22:
            w1[i] = 14
        else:
            w1[i] = 1

    w3 = []
    for i in range(n2):
        w3.append(math.exp(w1[i] * w2[i]))

    net = np.zeros((n1, n1))
    for i in range(len(dic0)):
        for j in range(len(dic0)):
            if xs(dic0[i][1], dic0[j][1], w3) >= 0 or xs(dic0[i][1], dic0[j][1], w3) <= 0:
                net[i][j] = xs(dic0[i][1], dic0[j][1], w3)
    for i in range(len(dic0)):
        net[i][i] = 0
    return net


def Target_node_degree_matrix(net):
    n = len(net)
    diag = np.zeros((n, n))
    for i in range(n):
        d1 = 0
        for j in range(n):
            d1 += net[i, j]
        diag[i, i] = d1
    # for i in range(n):
    #     for j in range(n):
    #         net[i, j] = net[i, j] / diag[i, i]
    #     diag[i, i] = diag[i, i] / diag[i, i]
    return diag


def alm(n, m, l, alpha, O, D):
    # 初始化矩阵
    P = np.random.rand(n, m)
    Y = P
    C = np.random.rand(l, n)
    for i in range(n):
        xx = 0
        for j in range(l):
            xx += C[j][i]
        for j in range(l):
            C[j][i] = C[j][i] / xx
    Q = C
    lambda1 = np.zeros((m, m))
    lambda2 = np.zeros((n, n))
    lambda3 = np.zeros((n, m))
    lambda4 = np.zeros((l, n))
    miu = 1e-6
    miu_max = 1e6
    rou = 1.01
    epsilon = 1e-8
    k = 0

    # 更新辅助矩阵
    B = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            b1 = 0
            b2 = 0
            for u in range(l):
                b1 += alpha * O[i, j] * C[u, i] * C[u, j]
                b2 += (C[u, i] - C[u, j]) * (C[u, i] - C[u, j]) / n
            B[i, j] = b1 - b2
    H = np.zeros((n, n))
    for i in range(n):
        h1 = 0
        for j in range(n):
            h1 += B[i, j]
        H[i, i] = h1
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            g1 = 0
            for u in range(m):
                g1 += (Y[i, u] - Y[j, u]) * (Y[i, u] - Y[j, u]) / n
            G[i, j] = g1
    J = np.zeros((n, n))
    for i in range(n):
        j1 = 0
        for j in range(n):
            j1 += G[i, j]
        J[i, i] = j1
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            w1 = 0
            for u in range(m):
                w1 += (Y[i, u] - Y[j, u]) * (Y[i, u] - Y[j, u]) * O[i, j]
            W[i, j] = w1

    # 计算拉格朗日函数
    L1 = np.trace(np.dot(np.dot(Y.T, (H - B)), Y)) + np.trace(
        np.dot(lambda1.T, (np.eye(m) - np.dot(np.dot(Y.T, D), P)))) + np.trace(
        np.dot(lambda2.T, (np.ones((n, n)) - np.dot(Q.T, np.ones((l, n)))))) + np.trace(
        np.dot(lambda3.T, (Y - P))) + np.trace(np.dot(lambda4.T, (C - Q))) + miu / 2 * (
                 np.linalg.norm((np.eye(m) - np.dot(np.dot(Y.T, D), P)), "fro") + np.linalg.norm(
        (np.ones((n, n)) - np.dot(Q.T, np.ones((l, n)))), "fro") + np.linalg.norm((Y - P), "fro")
                 + np.linalg.norm((C - Q), "fro"))
    L2 = 0

    # 循环
    while abs(L1 - L2) > epsilon:
        # 更新Y, P, C, Q
        Y0 = np.dot(np.linalg.inv(2 * H - 2 * B + miu * np.dot(np.dot(np.dot(D, P), P.T), D) + miu * np.eye(n)),
                    (np.dot(np.dot(D, P), lambda1.T) - lambda3 + miu * np.dot(D, P) + miu * P))
        P0 = np.dot(np.linalg.inv(miu * np.dot(np.dot(np.dot(D, Y0), Y0.T), D) + miu * np.eye(n)),
                    (np.dot(np.dot(D, Y0), lambda1) + lambda3 + miu * np.dot(D, Y0) + miu * Y0))
        C0 = np.dot((miu * Q - lambda4), np.linalg.inv(alpha * W - 2 * J + 2 * G + miu * np.eye(n)))
        Q0 = np.dot(np.linalg.inv(miu * n * np.ones((l, l)) + miu * np.eye(l)),
                    (np.dot(np.ones((l, n)), lambda2.T) + lambda4 + miu * n * np.ones((l, n)) + miu * C0))

        # 更新拉格朗日乘子
        lambda10 = lambda1 + miu * (np.eye(m) - np.dot(np.dot(Y0.T, D), P0))
        lambda20 = lambda2 + miu * (np.ones((n, n)) - np.dot(Q0.T, np.ones((l, n))))
        lambda30 = lambda3 + miu * (Y0 - P0)
        lambda40 = lambda4 + miu * (C0 - Q0)

        # 更新B, H, G, J, W
        B0 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                b1 = 0
                b2 = 0
                for u in range(l):
                    b1 += alpha * O[i, j] * Q0[u, i] * Q0[u, j]
                    b2 += (Q0[u, i] - Q0[u, j]) * (Q0[u, i] - Q0[u, j]) / n
                B0[i, j] = b1 - b2
        H0 = np.zeros((n, n))
        for i in range(n):
            h1 = 0
            for j in range(n):
                h1 += B0[i, j]
            H0[i, i] = h1
        G0 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                g1 = 0
                for u in range(m):
                    g1 += (Y0[i, u] - Y0[j, u]) * (Y0[i, u] - Y0[j, u]) / n
                G0[i, j] = g1
        J0 = np.zeros((n, n))
        for i in range(n):
            j1 = 0
            for j in range(n):
                j1 += G0[i, j]
            J0[i, i] = j1
        W0 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                w1 = 0
                for u in range(m):
                    w1 += (Y0[i, u] - Y0[j, u]) * (Y0[i, u] - Y0[j, u]) * O[i, j]
                W0[i, j] = w1

        # 检查收敛
        miu0 = min(miu * rou, miu_max)
        L2 = L1
        L1 = np.trace(np.dot(np.dot(Y0.T, (H0 - B0)), Y0)) + np.trace(
            np.dot(lambda10.T, (np.eye(m) - np.dot(np.dot(Y0.T, D), P0)))) + np.trace(
            np.dot(lambda20.T, (np.ones((n, n)) - np.dot(Q0.T, np.ones((l, n)))))) + np.trace(
            np.dot(lambda30.T, (Y0 - P0))) + np.trace(np.dot(lambda40.T, (C0 - Q0))) + miu0 / 2 * (
                     np.linalg.norm((np.eye(m) - np.dot(np.dot(Y0.T, D), P0)), "fro") + np.linalg.norm(
                 (np.ones((n, n)) - np.dot(Q0.T, np.ones((l, n)))), "fro") + np.linalg.norm((Y0 - P0), "fro")
                     + np.linalg.norm((C0 - Q0), "fro"))

        # 更新参数
        Y = Y0
        P = P0
        C = C0
        Q = Q0
        B = B0
        H = H0
        G = G0
        J = J0
        W = W0
        lambda1 = lambda10
        lambda2 = lambda20
        lambda3 = lambda30
        lambda4 = lambda40
        miu = miu0
        k += 1
        print(k, abs(L1 - L2))
        if k > 100:
            break
    return Y, C


if __name__ == '__main__':
    path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\vote_net50.txt'
    output1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\O.csv'
    output2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\D.csv'

    with open(path1, "r") as f:
        data = f.readlines()
    O1 = Target_node_weight(data)
    D = Target_node_degree_matrix(O1)
    mm = 10
    ll = 4
    alpha = 1
    # Y, C = alm(len(D), mm, ll, alpha, O1, D)
    df1 = pd.DataFrame(O1)
    df2 = pd.DataFrame(D)
    df1.to_csv(output1, index=False)
    df2.to_csv(output2, index=False)
    # fig = plt.figure(figsize=(12, 6))
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y)
    #
    # ax2 = fig.add_subplot(122)
    # ax2.scatter(X_ndim[:, 0], X_ndim[:, 1], c=Y)
    # plt.show()

    # example 2: hand-written digits
    # X = load_digits().data
    # y = load_digits().target
    #
    # dist = cal_pairwise_dist(X)
    # max_dist = np.max(dist)
    # print("max_dist", max_dist)
    # X_ndim = le(X, n_neighbors=20, t=max_dist*0.1)
    # plt.scatter(X_ndim[:, 0], X_ndim[:, 1], c=y)
    # plt.savefig("LE2.png")
    # plt.show()
