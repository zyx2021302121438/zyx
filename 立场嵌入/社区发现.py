import numpy as np
import pandas as pd
import json
import os
import networkx as nx
import pickle as pkl
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as col
from matplotlib import cm
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y


def run_kmeans_unlabeled(embeddings, k):
    labels_list = []
    sc_list = []
    for i in range(2, k + 1):
        estimator = KMeans(n_clusters=i)
        estimator.fit(embeddings)
        # y_pred = estimator.predict(embeddings)
        labels = estimator.labels_
        labels_list.append(labels)
        sc = metrics.silhouette_score(embeddings, labels, metric='euclidean')
        sc_list.append(sc)

    return labels_list, sc_list


def ch_score(X, labels):
    X, labels = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    n_samples, _ = X.shape
    n_labels = len(le.classes_)

    extra_disp, intra_disp = 0.0, 0.0
    mean = np.mean(X, axis=0)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0)
        extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
        intra_disp += np.sum((cluster_k - mean_k) ** 2)

    return extra_disp, intra_disp, extra_disp / intra_disp


def mod(partition, graph, weight='weight'):
    degn = {}
    degp = {}
    dn = 0.
    dp = 0.
    m = 0.
    for node in graph:
        degp[node] = 0.
        degn[node] = 0.
        for neighbor, datas in graph[node].items():
            if datas.get(weight, 0) == 1:
                degp[node] += 1
                dp += 1
            elif datas.get(weight, 0) == -1:
                degn[node] += 1
                dn += 1
    for node1 in graph:
        for node2 in graph:
            if partition[node2] == partition[node1]:
                if dn == 0:
                    m += graph[node1].get(node2, {}).get(weight, 0) - degp[node1] * degp[node2] / dp
                else:
                    m += graph[node1].get(node2, {}).get(weight, 0) - degp[node1] * degp[node2] / dp + degn[node1] * degn[node2] / dn
    res = m / (dn + dp)
    return res


def plot(data_2d, label, path, edges, node_number=False):
    plt.style.use('classic')
    # color_list = ['0, 98, 150', '183, 25, 30', '79, 164, 73']
    # color_list = [RGB_to_Hex(x) for x in color_list]
    color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    cmap2 = col.LinearSegmentedColormap.from_list('own2', color_list)
    # extra arguments are N=256, gamma=1.0
    cm.register_cmap(cmap=cmap2)
    # we can skip name here as it was already defined
    color_map = cm.get_cmap('own2')

    color_dict = dict(zip(range(len(color_list)), color_list))
    node_colors = [color_dict[i] for i in label]
    alpha = 0.5
    plt.figure(figsize=(20, 16))

    if edges is not None:
        for edge in edges:
            node1 = int(edge[0])  # 起点的索引
            node2 = int(edge[1])  # 终点的索引
            x1, y1 = data_2d[node1, 0], data_2d[node1, 1]  # 起点的坐标
            x2, y2 = data_2d[node2, 0], data_2d[node2, 1]  # 终点的坐标
            if edge[2] == '1':
                plt.plot([x1, x2], [y1, y2], color='green', linewidth=0.5, linestyle='-', alpha=0.2)
            else:
                plt.plot([x1, x2], [y1, y2], color='red', linewidth=0.5, linestyle='-', alpha=0.2)

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=node_colors, s=300, alpha=1)

    if node_number:
        for i in range(len(data_2d)):
            plt.text(data_2d[i, 0], data_2d[i, 1] - 0.02, i, fontsize=15, ha='center')  # 添加编号文本，可以调整偏移和字体大小

    # plt.xlim(-20, 80)
    # plt.ylim(-80, 60)
    # plt.axis('off')
    # plt.axis('equal')
    # plt.grid()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.show()


# path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\vote_nodes50.txt'
# sign = []
# node = []
# with open(path2, "r") as f:
#     data = f.readlines()
# for i in range(100):
#     sign.append(data[i].split('\t'))
# # for i in range(100, 114):
# #     nodeV.append(data[i].split('\t')[0])
# for i in sign:
#     if i[2] == '100\n':
#         node.append('民主党')
#     elif i[2] == '200\n':
#         node.append('共和党')
#     elif i[2] == '328\n':
#         node.append('独立')
#
path = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\投票\senate\Y3.csv'
df = pd.read_csv(path, header=None)
embeddings = []
edges = []
for j in range(len(df)):
    embeddings.append(list(df.loc[j]))
embedded = TSNE(n_components=2).fit_transform(embeddings)
# embedded = np.array(embeddings)
# path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\投票\house\house_xs_weight.txt'
# with open(path1, "r") as f:
#     data1 = f.readlines()
# edges = []
# for i in data1:
#     if i.split('\t')[2][:-1] > '0.5':
#         edges.append([str(int(i.split('\t')[0])-1), str(int(i.split('\t')[1])-1), '1'])
#     elif i.split('\t')[2][:-1] < '-0.99':
#         edges.append([str(int(i.split('\t')[0])-1), str(int(i.split('\t')[1])-1), '-1'])
# path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\MovieLens\MovieLens_sq.txt'
# with open(path2, "r") as f:
#     data2 = f.readlines()
# sign1 = []
# for i in data2:
#     sign1.append(int(i[:-1]))
# path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\投票\house\C2.csv'
# cf = pd.read_csv(path2, header=None)
sign1 = []
sign2 = []
# for i in range(len(cf)):
#     sign1.append(list(cf.loc[i]).index(max(list(cf.loc[i]))))
# for j in range(len(cf)):
#     sign1.append(list(cf.loc[j]))
# estimator = KMeans(n_clusters=2)
# estimator.fit(sign1)
# sign1 = estimator.labels_
path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\投票\senate\senate_nodes.txt'
with open(path2, "r") as f:
    data = f.readlines()
for i in range(len(data)):
    # sign2.append(float(data[i][:-1]))
    # sign2.append(round(float(data[i][:-1])))
    sign2.append(int(data[i].split('\t')[1][:-1]))
out_path1 = r'C:\Users\YHR\Desktop\house2.png'
plot(embedded, sign2, out_path1, edges, node_number=False)
# exs = []
# ins = []
# chs = []
# Q = []
# # N = 4
# for i in range(6, 7):
#     path = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\edge_list\sgcn\vote_xs50_0.' + str(i) + '0.csv'
#     path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\edge_list\vote_xs50_0.' + str(i) + '0.csv'
#     # out_path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\edge_list\图片\vote_xs50_0.' + str(i) + '04.png'
#     # out_path2 = r'C:\Users\YHR\Desktop\社区.xlsx'
#     df = pd.read_csv(path)
#     embeddings = []
#     edges = []
#     for j in range(len(df)):
#         embeddings.append(list(df.loc[j])[1:])
#     embedded = TSNE(n_components=2).fit_transform(embeddings)
#     label, scs = run_kmeans_unlabeled(embeddings, 15)
#     # plt.plot(range(2, 16), scs)
#     # plt.show()
#     G = nx.read_edgelist(path1, nodetype=str, delimiter=",", data=(("weight", float),))
#     with open(path1, "r") as f:
#         data = f.readlines()
#     for j in range(len(data)):
#         edges.append([data[j].split(',')[0], data[j].split(',')[1], data[j].split(',')[2][:-1]])
#     for N in range(2, 10):
#         labs = {}
#         for j in range(len(label[N - 2])):
#             labs[str(j)] = label[N - 2][j]
#
#         Q.append(mod(labs, G))
#
#         out_path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\edge_list\图片\vote_xs50_weight0.60' + str(N) + '.png'
#         # plot(embedded, label[N - 2], out_path1, edges=edges, node_number=True)
#
#         extra, intra, ch = ch_score(embeddings, label[N - 2])
#         exs.append(extra)
#         ins.append(intra)
#         chs.append(ch)
# print(Q)
# print(exs)
# print(ins)
# print(chs)


# data1 = pd.DataFrame(columns=['编号', '姓名', '党派', '社区'])
# for i in range(100):
#     data1.loc[len(data1.index)] = [sign[i][0], sign[i][1], node[i], label[5][i]]
# data1.to_excel(out_path2)
# dimensionality_reduction_plot(embeddings, y_pred, save_path)
#
#
# color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
# cmap2 = col.LinearSegmentedColormap.from_list('own2', color_list)
# # extra arguments are N=256, gamma=1.0
# cm.register_cmap(cmap=cmap2)
# # we can skip name here as it was already defined
# color_map = cm.get_cmap('own2')
# color_dict = dict(zip(range(len(color_list)), color_list))
#
# alpha = 0.5
# for i in range(7):
#     plt.scatter(i, i, color=color_list[i], s=150, alpha=alpha, label=str(i))
# plt.legend()
# plt.show()
