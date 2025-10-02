import numpy as np
import pandas as pd
import json
import os
import networkx as nx
import pickle as pkl
from textblob import TextBlob
from collections import Counter
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.colors as col
from matplotlib import cm
from sklearn import metrics


def plot(data_2d, label, path, edges, node_number=False):
    plt.style.use('classic')
    # color_list = ['0, 98, 150', '183, 25, 30', '79, 164, 73']
    # color_list = [RGB_to_Hex(x) for x in color_list]
    color_list = ['#e41a1c', '#FFFFFF', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999', '#808080']
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
            if edge[2] > 0:
                plt.plot([x1, x2], [y1, y2], color='green', linewidth=0.5, linestyle='-', alpha=0.2)
            else:
                plt.plot([x1, x2], [y1, y2], color='red', linewidth=0.5, linestyle='-', alpha=0.2)

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=node_colors, s=150, alpha=1)

    # if node_number:
    #     for i in range(len(data_2d)):
    #         plt.text(data_2d[i, 0], data_2d[i, 1] - 0.02, i, fontsize=15, ha='center')  # 添加编号文本，可以调整偏移和字体大小

    # plt.xlim(-20, 80)
    # plt.ylim(-80, 60)
    # plt.axis('off')
    # plt.axis('equal')
    # plt.grid()
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.show()


path = r'E:\毕业论文\数据集\美国大选1225\processed.csv'
path1 = r'E:\毕业论文\数据集\美国大选1225\mgdx0.txt'
path2 = r'E:\毕业论文\数据集\美国大选1225\mgdx1.txt'

# with open(path1, "r") as f:
#     data = f.readlines()
# data1 = (set(data))
# u = []
# for i in data1:
#     u.append([i.split('\t')[0]+i.split('\t')[1], i.split('\t')[2]])
# count_dict = {}
# for key, value in u:
#     if key in count_dict:
#         continue
#     else:
#         count_dict[key] = value
# f3 = open(path2, 'w', encoding='utf-8')
# for key in list(count_dict):
#     f3.writelines([key[0], '\t', key[1:], '\t', count_dict[key]])
# f3.close()

# with open(path3, "r") as f:
#     data = f.readlines()
# u = []
# for i in data:
#     u.append(i.split('\t')[1])
# print(len(set(u)))

# person = []
# for i in data:
#     person.append(i.split('\t')[1])
# counter = Counter(person)
# result_list = [item for item, count in counter.items() if count > 2][:1000]
# f2 = open(path2, 'w', encoding='utf-8')
# for i in data:
#     if i.split('\t')[1] in result_list:
#         f2.writelines(['100'+i.split('\t')[0], '\t', str(result_list.index(i.split('\t')[1])), '\t', i.split('\t')[2]])
# f2.close()
# u = []
# for i in data:
#     u.append(i.split('\t')[0]+i.split('\t')[1])
# print(len(set(u)))

# model1 = torch.load(r'E:\毕业论文\数据集\美国大选1225\result\tensor1_2.pt')
# model2 = torch.load(r'E:\毕业论文\数据集\美国大选1225\result\tensor2_2.pt')
path3 = r'E:\SignNetwork\SGCN-master\input\mgdx_4.csv'
path4 = r'E:\SignNetwork\SGCN-master\X.csv'
out_path1 = r'C:\Users\YHR\Desktop\mgdx.png'
# with torch.no_grad():
#     m1 = model1.numpy()
#     m2 = model2.numpy()
# m = np.zeros((1260, 32))
# m[:1256, :] = m2
# m[1256:, :] = m1
df = pd.read_csv(path4, header=None)
embeddings = []
dataset = pd.read_csv(path3).values.tolist()
edges = [edge[0:3] for edge in dataset]
for j in range(len(df)):
    embeddings.append(list(df.loc[j]))
embedded = TSNE(n_components=2).fit_transform(embeddings)
# embedded = TSNE(n_components=2).fit_transform(m)
# df = pd.DataFrame(embedded)
# df.to_csv(r'E:\毕业论文\数据集\美国大选1225\result\output2.csv', index=False)
# estimator = KMeans(n_clusters=2)
# estimator.fit(embedded)
# sign = estimator.labels_
sign = [0] * 4 + [1] * 1978
plot(embedded, sign, out_path1, edges=edges, node_number=True)
# labels_list = []
# sc_list = []
# for i in range(2, 15):
#     estimator = KMeans(n_clusters=i)
#     estimator.fit(embedded)
#     # y_pred = estimator.predict(embeddings)
#     labels = estimator.labels_
#     labels_list.append(labels)
#     sc = metrics.silhouette_score(embedded, labels, metric='euclidean')
#     sc_list.append(sc)
# plt.plot(range(2, 15), sc_list)
# plt.show()
