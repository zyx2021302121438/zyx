import numpy as np
import pandas as pd
import json
import os
import networkx as nx
import pickle as pkl
import jieba
from snownlp import SnowNLP
from collections import Counter
import matplotlib.pyplot as plt
import random


# path = r'E:\毕业论文\数据集\哇哈哈\2024.2-4 Wahaha.xlsx'
# path1 = r'E:\毕业论文\数据集\哇哈哈\Wahaha.xlsx'
# data = pd.read_excel(path)
# topic = []
# df = pd.DataFrame(columns=['user_id', 'content', 'ref_url'])
# for i in range(len(data)):
#     if type(data.loc[i]['content']) is str and type(data.loc[i]['ref_url']) is str and '#' in data.loc[i]['content']:
#         df.loc[len(df.index)] = [str(int(float(data.loc[i]['user_id']))), str(data.loc[i]['content']), data.loc[i]['ref_url']]
# df.to_excel(path1, index=False)


# path1 = r'E:\毕业论文\数据集\哇哈哈\Wahaha.xlsx'
# path2 = r'E:\毕业论文\数据集\哇哈哈\Wahaha1.xlsx'
# data = pd.read_excel(path1)
# topic = []
# df = pd.DataFrame(columns=['user_id', 'content', 'ref_url'])
# for i in range(len(data)):
#     print(i)
#     tmp = data.loc[i]['content'].split('#')
#     for j in range(int((len(tmp)-1)/2)):
#         if '宗庆后' in tmp[j * 2 + 1] and len(tmp[j * 2 + 1]) < 50:
#             topic.append(tmp[j * 2 + 1])
# counter = Counter(topic)
# result = [item for item, count in counter.items() if count > 100]
# for i in range(len(data)):
#     print(i)
#     tmp = data.loc[i]['content'].split('#')
#     for j in range(int((len(tmp)-1)/2)):
#         if tmp[j * 2 + 1] in result:
#             df.loc[len(df.index)] = [str(int(float(data.loc[i]['user_id']))), str(data.loc[i]['content']), data.loc[i]['ref_url']]
#             break
# df.to_excel(path2, index=False)

path2 = r'E:\毕业论文\数据集\哇哈哈\Wahaha1.xlsx'
path3 = r'E:\毕业论文\数据集\哇哈哈\Wahaha2.xlsx'
# data = pd.read_excel(path2)
# topic = []
# df = pd.DataFrame(columns=['user_id', 'content', 'ref_url', 'topic'])
# for i in range(len(data)):
#     print(i)
#     tmp = data.loc[i]['content'].split('#')
#     for j in range(int((len(tmp)-1)/2)):
#         if '宗庆后' in tmp[j * 2 + 1] and len(tmp[j * 2 + 1]) < 50:
#             topic.append(tmp[j * 2 + 1])
# counter = Counter(topic)
# result = [item for item, count in counter.items() if count > 100]
# print(len(result))
# print(result)
# for i in range(len(data)):
#     print(i)
#     tmp = data.loc[i]['content'].split('#')
#     for j in range(int((len(tmp)-1)/2)):
#         if tmp[j * 2 + 1] in result:
#             df.loc[len(df.index)] = [str(int(float(data.loc[i]['user_id']))), str(data.loc[i]['content']), data.loc[i]['ref_url'].split('/')[-2], str(result.index(tmp[j * 2 + 1]))]
#             break
# df.to_excel(path3, index=False)


'''情感抽取'''
path4 = r'E:\毕业论文\数据集\哇哈哈\Wahaha_st.csv'
path5 = r'E:\毕业论文\数据集\哇哈哈\Wahaha_zf.csv'
# data = pd.read_excel(path3)
# df1 = pd.DataFrame(columns=['from', 'to', 'sign'])
# df2 = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(data)):
#     print(i)
#     if '[rtt]' in data.loc[i]['content']:
#         text1 = data.loc[i]['content'].split('[rtt]')[0]
#         text2 = data.loc[i]['content'].split('[rtt]')[1]
#     else:
#         text1 = data.loc[i]['content']
#         text2 = data.loc[i]['content']
#
#     if text1 == '转发微博':
#         polarity1 = 1
#     else:
#         words1 = jieba.lcut(text1)
#         sentence1 = ' '.join(words1)
#         polarity1 = SnowNLP(sentence1).sentiments * 2 - 1
#
#     words2 = jieba.lcut(text2)
#     sentence2 = ' '.join(words2)
#     polarity2 = SnowNLP(sentence2).sentiments * 2 - 1
#
#     df1.loc[len(df1.index)] = [str(int(float(data.loc[i]['ref_url']))),
#                                str(int(float(data.loc[i]['user_id']))),
#                                float(polarity1) * float(polarity2)]
#     df2.loc[len(df2.index)] = [str(data.loc[i]['topic']), str(int(float(data.loc[i]['ref_url']))), float(polarity2)]
#     df2.loc[len(df2.index)] = [str(data.loc[i]['topic']), str(int(float(data.loc[i]['user_id']))), float(polarity1)]
#
# df1.to_csv(path5, index=False)
# st = df2.groupby(['from', 'to'])['sign'].apply(lambda x: x.mean()).reset_index()
# st.to_csv(path4, index=False)


'''用户节点'''
path6 = r'E:\毕业论文\数据集\哇哈哈\user_0.txt'
# data = pd.read_csv(path4)
# user = list(data['to'])
# counter = Counter(user)
# result_list = [item for item, count in counter.items() if count > 1]
# f3 = open(path6, 'w', encoding='utf-8')
# for i in result_list:
#     f3.writelines([str(int(i)), '\n'])
# f3.close()


'''转发网络生成'''
# with open(path6, "r") as f:
#     user = f.readlines()
# user1 = []
# for i in user:
#     user1.append(i[:-1])
# path7 = r'E:\毕业论文\数据集\哇哈哈\Wahaha_zf1.csv'
# data = pd.read_csv(path5)
# data = data.groupby(['from', 'to'])['sign'].apply(lambda x: x.mean()).reset_index()
# df1 = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(data)):
#     if str(int(data.loc[i]['from'])) in user1 and str(int(data.loc[i]['to'])) in user1:
#         df1.loc[len(df1.index)] = list(data.loc[i])
# st = pd.read_csv(path4)
# user2 = []
# for i in range(len(df1)):
#     user2.append(df1.loc[i]['from'])
#     user2.append(df1.loc[i]['to'])
# user2 = list(set(user2))
# print(len(user2))
# print(len(df1))
# df2 = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(df1)):
#     if float(df1.loc[i]['sign']) >= 0:
#         df2.loc[len(df2.index)] = [str(user2.index(df1.loc[i]['from'])), str(user2.index(df1.loc[i]['to'])), '1']
#     elif float(df1.loc[i]['sign']) < 0:
#         df2.loc[len(df2.index)] = [str(user2.index(df1.loc[i]['from'])), str(user2.index(df1.loc[i]['to'])), '-1']
# df2.to_csv(path7, index=False)
# df3 = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(st)):
#     if st.loc[i]['to'] in user2:
#         df3.loc[len(df3.index)] = [str(int(st.loc[i]['from']) + len(user2)), str(user2.index(st.loc[i]['to'])), st.loc[i]['sign']]
# path8 = r'E:\毕业论文\数据集\哇哈哈\Wahaha_st1.csv'
# df3.to_csv(path8, index=False)


'''数据清理'''
path10 = r'E:\毕业论文\数据集\哇哈哈\Wahaha_st1.csv'
path11 = r'E:\毕业论文\数据集\哇哈哈\Wahaha_st2.csv'
data = pd.read_csv(path10)
user = list(set(list(data['from'])))
num = len(set(list(data['to'])))
df3 = pd.DataFrame(columns=['from', 'to', 'sign'])
for i in range(len(data)):
    df3.loc[len(df3.index)] = [str(user.index(data.loc[i]['from']) + num), data.loc[i]['to'], data.loc[i]['sign']]
df3.to_csv(path11, index=False)


'''平衡结构统计'''
# with open(path2, "r") as f:
#     data = f.readlines()
# N = 58793
# M = 5
# A=0
# B=0
# C=0
# D=0
# E=0
# dic = np.zeros((M, N))
# for i in data:
#     # dic.setdefault(int(i.split('\t')[0]), [])
#     # dic[int(i.split('\t')[0])].append(fh(i.split('\t')[2][:-1]))
#     dic[int(i.split('\t')[0]), int(i.split('\t')[1])] = int(i.split('\t')[2][:-1])
# for k in range(M):
#     for v in range(M):
#         if v == k:
#             continue
#         else:
#             for i in range(N - 1):
#                 if dic[k, i] == 0 or dic[v, i] == 0:
#                     continue
#                 else:
#                     for j in range(i + 1, N):
#                         if dic[k, j] == 0 or dic[v, j] == 0:
#                             continue
#                         else:
#                             die = [dic[k, i], dic[k, j], dic[v, j], dic[v, i]]
#                             # if die == [1, 1, 1, 1]:
#                             #     A += 1
#                             # elif die == [-1, -1, -1, -1]:
#                             #     B += 1
#                             # elif die.count(-1) == 2 and die.count(1) == 2:
#                             #     C += 1
#                             # elif die.count(-1) == 1 and die.count(1) == 3:
#                             #     D += 1
#                             # elif die.count(-1) == 3 and die.count(1) == 1:
#                             #     E += 1
#                             if die.count(-1) == 1 or die.count(-1) == 3:
#                                 if k==0 or v==0:
#                                     A+=1
#                                 if k==1 or v==1:
#                                     B+=1
#                                 if k==2 or v==2:
#                                     C+=1
#                                 if k==3 or v==3:
#                                     D+=1
#                                 if k==4 or v==4:
#                                     E+=1
#     # print(k)
#     # print(A + B + C)
#     # print(D + E)
# print(A,B,C,D,E)

'''画图'''
# path9 = r'E:\毕业论文\数据集\微博吴亦凡\wyf_zf111.csv'
# a = nx.read_edgelist(path9, delimiter=',', nodetype=int, data=(("weight", float),))
# # a = list(G.subgraph(c) for c in nx.connected_components(G))[0]
# # G.remove_node(3)
# noden = range(7539)
# edgep = [(u, v) for (u, v, d) in a.edges(data=True) if d['weight'] == 1]
# edgen = [(u, v) for (u, v, d) in a.edges(data=True) if d['weight'] == -1]
# print(len(edgep), len(edgen))
# pos = nx.spring_layout(a)
# # print(pos)
# # nx.draw(a, pos, node_size=10, node_color='black')
# # nx.draw_networkx_nodes(a, pos, nodelist=nodep, node_size=10, node_color='b')
# nx.draw_networkx_nodes(a, pos, nodelist=noden, node_size=10, node_color='y')
# nx.draw_networkx_edges(a, pos, edgelist=edgep, width=0.1, edge_color='g')
# nx.draw_networkx_edges(a, pos, edgelist=edgen, width=0.2, edge_color='r')
# plt.show()

'''数据集格式转换'''
# for k in range(5):
#     path9 = r'E:\毕业论文\数据集\微博吴亦凡\wyf_st' + str(k + 1) + '.csv'
#     path10 = r'E:\毕业论文\数据集\微博吴亦凡\wyf_st1' + str(k + 1) + '.csv'
#     data = pd.read_csv(path9)
#     usernum = len(set(list(data['to'])))
#     print(usernum)
#     df = pd.DataFrame(columns=['from', 'to', 'sign'])
#     for i in range(len(data)):
#         df.loc[len(df.index)] = [str(int(data.loc[i]['from'])+usernum), data.loc[i]['to'], data.loc[i]['sign']]
#     df.to_csv(path10, index=False)
