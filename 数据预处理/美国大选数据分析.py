import numpy as np
import pandas as pd
import json
import os
import networkx as nx
import pickle as pkl
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
import random


path = r'E:\毕业论文\数据集\美国大选1225\processed.csv'

'''情感抽取'''
# path2 = r'E:\毕业论文\数据集\美国大选1225\mgdx_st.txt'
# data = pd.read_csv(path, low_memory=False)
# df = pd.DataFrame(columns=['source', 'target', 'sign'])
# for i in range(len(data)):
#     if type(data.loc[i]['content']) is str:
#         if '[cmt]' in data.loc[i]['content']:
#             text1 = data.loc[i]['content'].split('[cmt]')[0]
#             text2 = data.loc[i]['content'].split('[cmt]')[1]
#         else:
#             text1 = data.loc[i]['content']
#             text2 = data.loc[i]['content']
#
#         if 'Biden' in text1 or 'biden' in text1:
#             blob = TextBlob(text1)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df.loc[len(df.index)] = ['0', str(int(float(data.loc[i]['article_id']))), float(polarity)]
#
#         elif 'Trump' in text1 or 'trump' in text1:
#             blob = TextBlob(text1)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df.loc[len(df.index)] = ['1', str(int(float(data.loc[i]['article_id']))), float(polarity)]
#
#         elif 'Harris' in text1 or 'harris' in text1:
#             blob = TextBlob(text1)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df.loc[len(df.index)] = ['2', str(int(float(data.loc[i]['article_id']))), float(polarity)]
#
#         elif 'Vance' in text1 or 'vance' in text1:
#             blob = TextBlob(text1)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df.loc[len(df.index)] = ['3', str(int(float(data.loc[i]['article_id']))), float(polarity)]
#
#         if 'Biden' in text2 or 'biden' in text2:
#             blob = TextBlob(text2)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df.loc[len(df.index)] = ['0', str(int(float(data.loc[i]['root_article_id']))), float(polarity)]
#
#         elif 'Trump' in text2 or 'trump' in text2:
#             blob = TextBlob(text2)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df.loc[len(df.index)] = ['1', str(int(float(data.loc[i]['root_article_id']))), float(polarity)]
#
#         elif 'Harris' in text2 or 'harris' in text2:
#             blob = TextBlob(text2)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df.loc[len(df.index)] = ['2', str(int(float(data.loc[i]['root_article_id']))), float(polarity)]
#
#         elif 'Vance' in text2 or 'vance' in text2:
#             blob = TextBlob(text2)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df.loc[len(df.index)] = ['3', str(int(float(data.loc[i]['root_article_id']))), float(polarity)]
# result = df.groupby(['source', 'target'])['sign'].apply(lambda x: x.mean()).reset_index()
# f2 = open(path2, 'w', encoding='utf-8')
# for i in range(len(result)):
#     f2.writelines([result.loc[i]['source'], '\t', result.loc[i]['target'], '\t', str(result.loc[i]['sign']), '\n'])
# f2.close()


'''数据清理'''
# path2 = r'E:\毕业论文\数据集\美国大选1225\mgdx_st.txt'
# path4 = r'E:\毕业论文\数据集\美国大选1225\user_0.txt'
# with open(path2, "r") as f:
#     data = f.readlines()
# data1 = []
# for i in data:
#     if float(i.split('\t')[2][:-1]) > 0:
#         data1.append(i.split('\t')[0]+'\t'+i.split('\t')[1]+'\t1\n')
#     elif float(i.split('\t')[2][:-1]) < 0:
#         data1.append(i.split('\t')[0]+'\t'+i.split('\t')[1]+'\t-1\n')
# person = []
# for i in data1:
#     person.append(i.split('\t')[1])
# counter = Counter(person)
# result_list = [item for item, count in counter.items() if count > 1]
# f3 = open(path4, 'w', encoding='utf-8')
# for i in result_list:
#     f3.writelines([i, '\n'])
# f3.close()

'''转发网络提取'''
# path4 = r'E:\毕业论文\数据集\美国大选1225\user_0.txt'
# path5 = r'E:\毕业论文\数据集\美国大选1225\mgdx_zf_0.csv'
# path6 = r'E:\毕业论文\数据集\美国大选1225\user_1.txt'
# data = pd.read_csv(path, low_memory=False)
# with open(path4, "r") as f:
#     user = f.readlines()
# user1 = []
# for i in user:
#     user1.append(i[:-1])
# df1 = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(data)):
#     print(i)
#     if str(data.loc[i]['comment_count']) != 'nan':
#         if type(data.loc[i]['content']) is str and str(int(float(data.loc[i]['article_id']))) in user1:
#             if '[cmt]' in data.loc[i]['content']:
#                 text = data.loc[i]['content'].split('[cmt]')[0]
#             else:
#                 text = data.loc[i]['content']
#             blob = TextBlob(text)
#             polarity = blob.sentiment.polarity
#             if float(polarity) != 0.0:
#                 df1.loc[len(df1.index)] = [str(int(float(data.loc[i]['root_article_id']))), str(int(float(data.loc[i]['article_id']))), float(polarity)]
# df2 = df1.groupby(['from', 'to'])['sign'].apply(lambda x: x.mean()).reset_index()
# positive_edges = []
# negative_edges = []
# for i in range(len(df2)):
#     if df2.loc[i]['sign'] > 0:
#         positive_edges.append(list(df2.loc[i]))
#     else:
#         negative_edges.append(list(df2.loc[i]))
# pos_count = len(positive_edges)
# neg_count = len(negative_edges)
# remove_count = pos_count - int(neg_count * 1.5)
# if remove_count > 0:
#     keep_positive = random.sample(positive_edges, int(neg_count * 1.5))
#     balanced_data = keep_positive + negative_edges
# else:
#     balanced_data = positive_edges + negative_edges
# user = []
# for i in balanced_data:
#     user.append(i[0])
#     user.append(i[1])
# user = list(set(user))
# df = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in balanced_data:
#     if float(i[2]) > 0:
#         df.loc[len(df.index)] = [str(user.index(i[0])), str(user.index(i[1])), '1']
#     else:
#         df.loc[len(df.index)] = [str(user.index(i[0])), str(user.index(i[1])), '-1']
# for i in range(len(df)):
#     if int(df.loc[i]['from']) > int(df.loc[i]['to']):
#         tmp = int(df.loc[i]['from'])
#         df.loc[i]['from'] = int(df.loc[i]['to'])
#         df.loc[i]['to'] = tmp
# df.to_csv(path5)
# f3 = open(path6, 'w', encoding='utf-8')
# for i in user:
#     f3.writelines([i, '\n'])
# f3.close()


'''实体网络补全'''
# path6 = r'E:\毕业论文\数据集\美国大选1225\user_1.txt'
# path7 = r'E:\毕业论文\数据集\美国大选1225\mgdx_st_1.csv'
# data = pd.read_csv(path, low_memory=False)
# with open(path6, "r") as f:
#     user = f.readlines()
# user1 = []
# for i in user:
#     user1.append(i[:-1])
# df = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(data)):
#     print(i)
#     if str(data.loc[i]['comment_count']) != 'nan':
#         if type(data.loc[i]['content']) is str and str(int(float(data.loc[i]['article_id']))) in user1:
#             if '[cmt]' in data.loc[i]['content']:
#                 text = data.loc[i]['content'].split('[cmt]')[0]
#             else:
#                 text = data.loc[i]['content']
#
#             if 'Biden' in text or 'biden' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = [str(len(user1)), str(user1.index(str(int(float(data.loc[i]['article_id']))))), float(polarity)]
#
#             elif 'Trump' in text or 'trump' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = [str(len(user1)+1), str(user1.index(str(int(float(data.loc[i]['article_id']))))), float(polarity)]
#
#             elif 'Harris' in text or 'harris' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = [str(len(user1)+2), str(user1.index(str(int(float(data.loc[i]['article_id']))))), float(polarity)]
#
#             elif 'Vance' in text or 'vance' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = [str(len(user1)+3), str(user1.index(str(int(float(data.loc[i]['article_id']))))), float(polarity)]
#
#         if type(data.loc[i]['content']) is str and str(int(float(data.loc[i]['root_article_id']))) in user1:
#             if '[cmt]' in data.loc[i]['content']:
#                 text = data.loc[i]['content'].split('[cmt]')[1]
#             else:
#                 text = data.loc[i]['content']
#
#             if 'Biden' in text or 'biden' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = [str(len(user1)), str(user1.index(str(int(float(data.loc[i]['root_article_id']))))), float(polarity)]
#
#             elif 'Trump' in text or 'trump' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = [str(len(user1)+1), str(user1.index(str(int(float(data.loc[i]['root_article_id']))))), float(polarity)]
#
#             elif 'Harris' in text or 'harris' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = [str(len(user1)+2), str(user1.index(str(int(float(data.loc[i]['root_article_id']))))), float(polarity)]
#
#             elif 'Vance' in text or 'vance' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = [str(len(user1)+3), str(user1.index(str(int(float(data.loc[i]['root_article_id']))))), float(polarity)]
# result = df.groupby(['from', 'to'])['sign'].apply(lambda x: x.mean()).reset_index()
# for i in range(len(user1)):
#     if str(i) not in list(result['to']):
#         print(i)
#         result.loc[len(result.index)] = [str(len(user1)+random.randint(0, 3)), str(i), random.random()*2-1]
# result.to_csv(path7)

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
# path8 = r'E:\毕业论文\数据集\美国大选1225\mgdx_zf_1.csv'
# # path9 = r'E:\毕业论文\数据集\美国大选1225\mgdx_zf_3.csv'
# a = nx.read_edgelist(path8, delimiter=',', nodetype=int, data=(("weight", float),))
# # a = list(G.subgraph(c) for c in nx.connected_components(G))[0]
# # # G.remove_node(3)
# noden = range(1978)
# edgep = [[u, v] for (u, v, d) in a.edges(data=True) if d['weight'] == 1]
# edgen = [[u, v] for (u, v, d) in a.edges(data=True) if d['weight'] == -1]
# # edges = edgep + edgen
# # df1 = pd.DataFrame(columns=['from', 'to', 'sign'])
# # for i in range(len(edges)):
# #     df1.loc[len(df1.index)] = [edges[i][0], edges[i][1], edges[i][2]]
# # df1.to_csv(path9)
# print(len(edgep), len(edgen))
# pos = nx.spring_layout(a)
# # print(pos)
# # nx.draw(a, pos, node_size=10, node_color='black')
# # nx.draw_networkx_nodes(a, pos, nodelist=nodep, node_size=10, node_color='b')
# nx.draw_networkx_nodes(a, pos, nodelist=noden, node_size=10, node_color='y')
# nx.draw_networkx_edges(a, pos, edgelist=edgep, width=0.1, edge_color='g')
# nx.draw_networkx_edges(a, pos, edgelist=edgen, width=0.2, edge_color='r')
# plt.show()

'''公开数据集格式转换'''
# path9 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\投票\senate\senate.txt'
# path10 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\投票\senate\senate.csv'
# with open(path9, "r") as f:
#     data = f.readlines()
# # f2 = open(path3, 'w', encoding='utf-8')
# # for i in data1:
# #     if i.split('\t')[1] in result_list:
# #         f2.writelines([i.split('\t')[0], '\t', str(result_list.index(i.split('\t')[1])), '\t', i.split('\t')[2]])
# df = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in data:
#     df.loc[len(df.index)] = [i.split('\t')[0], i.split('\t')[1], i.split('\t')[2][:-1]]
# df.to_csv(path10)


'''草稿'''
path9 = r'E:\毕业论文\数据集\美国大选1225\mgdx_zf.csv'
data = pd.read_csv(path9)
df = pd.DataFrame(columns=['from', 'to', 'sign'])
x = 0
for i in range(len(data)):
    if data.loc[len(data)-i-1]['sign'] == 1 and x < 10000:
        df.loc[len(df.index)] = [data.loc[len(data)-i-1]['from'], data.loc[len(data)-i-1]['to'], data.loc[len(data)-i-1]['sign']]
        x += 1
    elif data.loc[len(data)-i-1]['sign'] == -1:
        df.loc[len(df.index)] = [data.loc[len(data)-i-1]['from'], data.loc[len(data)-i-1]['to'], data.loc[len(data)-i-1]['sign']]

path10 = r'E:\毕业论文\数据集\美国大选1225\mgdx_st_0.csv'
user = list(set(list(df['from']) + list(df['to'])))
df1 = pd.DataFrame(columns=['from', 'to', 'sign'])
for i in range(len(df)):
    df1.loc[len(df1.index)] = [str(user.index(df.loc[i]['from'])), str(user.index(df.loc[i]['to'])), df.loc[i]['sign']]
data1 = pd.read_csv(path10)
df2 = pd.DataFrame(columns=['from', 'to', 'sign'])
for i in range(len(data1)):
    if int(data1.loc[i]['to']) in user:
        df2.loc[len(df2.index)] = [str(data1.loc[i]['from']+len(user)), str(user.index(int(data1.loc[i]['to']))), data1.loc[i]['sign']]
path11 = r'E:\毕业论文\数据集\美国大选1225\mgdx_zf_2.csv'
path12 = r'E:\毕业论文\数据集\美国大选1225\mgdx_st_2.csv'
df1.to_csv(path11, index=False)
df2.to_csv(path12, index=False)
