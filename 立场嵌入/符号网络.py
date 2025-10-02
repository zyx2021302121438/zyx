import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from numpy import array
import os


def sign(x):
    y = 0
    if x == 'support':
        y = 1
    elif x == 'oppose':
        y = -1
    return y


path0 = r'C:\Users\YHR\Desktop\YHR\硕士\知识图谱\数据\politician_0517\politician_json'
path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\堕胎案新闻数据\news_list2.xlsx'
path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\堕胎案新闻数据\nodes2.txt'
path3 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\堕胎案新闻数据\edges2.txt'
path4 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\堕胎案新闻数据\edges20.txt'
path5 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\堕胎案新闻数据\sign20.txt'

'''判断节点极性'''
# with open(path3, "r") as f:
#     data1 = f.readlines()
# with open(path4, "r") as f:
#     data2 = f.readlines()
# edge1 = []
# edge2 = []
# node = []
# sign = []
# for i in range(len(data1)):
#     edge1.append(data1[i].split('\t'))
# for i in range(len(data2)):
#     edge2.append(data2[i].split('\t'))
# for i in edge2:
#     node.append(i[0])
#     node.append(i[1])
# node = list(set(node))
# for i in edge1:
#     if i[0] == '3' and i[1] in node:
#         sign.append(i)
#     if i[1] == '3' and i[0] in node:
#         sign.append(i)
# for i in sign:
#     data2.append(i[0] + '\t' + i[1] + '\t' + i[2])
# f3 = open(path5, 'w', encoding='utf-8')
# for i in data2:
#     f3.writelines(i)
# f3.close()

'''数据处理-筛选堕胎立场'''
# df = pd.read_excel(path1)
# abortion = []
# net = []
# net0 = []
# node = []
# edge = []
# for i in range(len(df)):
#     if df.iloc[i, 2] == 'abortion':
#         abortion.append(list(df.loc[i]))
# name = []
# names = []
# for i in abortion:
#     name.append(i[0])
#     names.append([i[0], sign(i[4])])
# for i in range(len(df)):
#     if df.iloc[i, 4] == 'oppose' or df.iloc[i, 4] == 'support':
#         net.append(list(df.loc[i]))
# for i in net:
#     if i[0] in name and i[2] in name:
#         node.append(i[0])
#         node.append(i[2])
#         net0.append(i)
# node = list(set(node))
# for i in net0:
#     edge.append([node.index(i[0]), node.index(i[2]), sign(i[4])])
# f1 = open(path2, 'w', encoding='utf-8')
# for i in node:
#     f1.writelines([str(node.index(i)), '\t', i, '\n'])
# f1.close()
# f2 = open(path3, 'w', encoding='utf-8')
# for i in edge:
#     f2.writelines([str(i[0]), '\t', str(i[1]), '\t', str(i[2]), '\n'])
# f2.close()
# f3 = open(path4, 'w', encoding='utf-8')
# for i in names:
#     f3.writelines([str(i[0]), '\t', str(i[1]), '\n'])
# f3.close()

'''构建符号网络'''
with open(path5, "r") as f:
    data = f.readlines()
nodep = []
noden = []
sign1 = []
for i in range(len(data)):
    sign1.append(data[i].split('\t'))
for i in sign1:
    if i[2] == '-1\n':
        noden.append(int(i[0]))
    elif i[2] == '1\n':
        nodep.append(int(i[0]))
nodep = list(set(nodep))
noden = list(set(noden))
G = nx.read_edgelist(path4, nodetype=int, data=(("weight", float),))
a = list(G.subgraph(c) for c in nx.connected_components(G))[0]
# G.remove_node(3)
# a = list(G.subgraph(c) for c in nx.connected_components(G))[0]
print(a)
edgep = [(u, v) for (u, v, d) in a.edges(data=True) if d['weight'] == 1]
edgen = [(u, v) for (u, v, d) in a.edges(data=True) if d['weight'] == -1]
print(len(edgep), len(edgen))
print(len(nodep), len(noden))
# pos = {2: (0, 0), 4: (0, 1), 5: (0, 2), 7: (0, 3), 8: (0, 4), 9: (0, 5), 10: (1, 0), 12: (1, 1), 14: (1, 2), 15: (1, 3),
#        16: (1, 4), 17: (1, 5), 18: (2, 0), 19: (2, 1), 20: (2, 2), 25: (2, 3), 26: (2, 4), 27: (2, 5), 28: (3, 1),
#        29: (3, 2), 30: (3, 3), 32: (3, 4), 33: (3, 5), 0: (4, 1), 1: (4, 2), 3: (4, 3), 6: (4, 4), 11: (4, 5),
#        13: (5, 1), 21: (5, 2), 22: (5, 3), 23: (5, 4), 24: (5, 5), 31: (5, 0)}
# pos = nx.spring_layout(a)
pos = {40: array([0.75762178, 0.1333895]), 181: array([0.7187887 , 0.11387628]), 50: array([0.13990592, 0.99532042]), 83: array([0.09190592, -0.74932042]), 173: array([0.07590592, -0.88332042]), 163: array([0.04690592, 0.81732042]), 154: array([-0.81769682, -0.12637809]), 28: array([-0.69660593, -0.05796838]),  23: array([0.63890592, 0.44532042]), 0: array([0.52376756, 0.36605342]), 127: array([0.49240289, 0.50934159]), 43: array([0.78513993, 0.4036502 ]), 177: array([0.5616381 , 0.39206766]), 15: array([-0.75634397, -0.63433353]), 122: array([0.44333936, 0.68862653]), 114: array([-0.72966496, -0.93635752]), 60: array([-0.69438865, -0.69270397]), 101: array([0.57390297, 0.29733307]), 94: array([ 0.62894197, -0.3076845 ]), 150: array([-0.06907837,  0.75250359]), 128: array([ 0.02657029, -0.76578729]), 21: array([ 0.61237944,  0.62628213]), 182: array([-0.91148479, -0.53296636]), 110: array([-0.8504197 , -0.37845761]), 59: array([ 0.51806735, -0.30847563]), 49: array([ 0.13890036, -0.42907018]), 19: array([0.39833654, 0.75304886]), 166: array([0.50801106, 0.7357439 ]), 14: array([0.69853232, 0.75603179]), 168: array([0.67553948, 0.502294  ]), 84: array([-0.97162825, -0.03454705]), 104: array([-0.7566517, -0.7132393]), 62: array([-0.97718056,  0.03438423]), 20: array([ 0.50202819, -0.05874402]), 138: array([-0.3217333 , -0.91985251]), 66: array([-0.30403627, -0.11827634]), 44: array([-0.74493651, -0.85774593]), 144: array([-0.89013518, -0.6402288 ]), 85: array([ 0.01287687,  0.13292721]), 195: array([-0.76859446, -0.52874476]), 145: array([0.84547739, 0.20358372]), 132: array([ 0.2811879 ,  0.74092106]), 172: array([-0.62906566, -0.99801854]), 54: array([-0.33422106, -0.81470398]), 107: array([0.17926571, 0.77776094]), 37: array([-0.76517408, -0.59083561]), 5: array([-0.74687303, -0.68388481]), 169: array([0.65060777, 0.10929411]), 91: array([0.30565864, 0.856016  ]), 38: array([0.59827863, 0.61497886]), 12: array([-0.36049413, -0.85894789]), 137: array([ 0.62580244, -0.36612754]), 22: array([0.38637156, 0.11469508]), 25: array([ 0.44341235,  0.44347414])}
# print(pos)
nx.draw(a, pos, node_size=50, node_color='black')
nx.draw_networkx_nodes(a, pos, nodelist=nodep, node_size=50, node_color='b')
nx.draw_networkx_nodes(a, pos, nodelist=noden, node_size=50, node_color='y')
nx.draw_networkx_edges(a, pos, edgelist=edgep, width=0.5, edge_color='g')
nx.draw_networkx_edges(a, pos, edgelist=edgen, width=0.5, edge_color='r')
plt.show()

'''数据处理-筛选公务员'''
# df = pd.read_excel(path1)
# net = []
# net0 = []
# node = []
# edge = []
# name = []
# # for root, dirs, files in os.walk(path0):
# #     for file in files:
# #         name.append(file.split('.')[0])
# for i in range(len(df)):
#     if df.iloc[i, 4] == 'oppose' or df.iloc[i, 4] == 'support':
#         net.append(list(df.loc[i]))
# for i in net:
#     # if i[1] == 'Person' and i[3] == 'Person':
#         node.append(i[0])
#         node.append(i[2])
#         net0.append(i)
# node = list(set(node))
# for i in net0:
#     edge.append([node.index(i[0]), node.index(i[2]), sign(i[4])])
# f1 = open(path2, 'w', encoding='utf-8')
# for i in node:
#     f1.writelines([str(node.index(i)), '\t', i, '\n'])
# f1.close()
# f2 = open(path3, 'w', encoding='utf-8')
# for i in edge:
#     f2.writelines([str(i[0]), '\t', str(i[1]), '\t', str(i[2]), '\n'])
# f2.close()
