import numpy as np
import pandas as pd
import json
import os
import networkx as nx
import pickle as pkl
from collections import Counter
import matplotlib.pyplot as plt
import random
import jieba
from snownlp import SnowNLP
from wordcloud import WordCloud
import matplotlib.font_manager as fm


path1 = r'E:\毕业论文\数据集\台湾选举\twitter_post.xlsx'
path2 = r'E:\毕业论文\数据集\台湾选举\twitter_comment.xlsx'


'''数据清理'''
path3 = r'E:\毕业论文\数据集\台湾选举\twitter1.xlsx'
path4 = r'E:\毕业论文\数据集\台湾选举\twitter2.xlsx'
# df1 = pd.DataFrame(columns=['user_id', 'ref_id', 'content'])
# data1 = pd.read_excel(path1)
# for i in range(len(data1)):
#     print(i)
#     if type(data1.loc[i]['content']) is str:
#         df1.loc[len(df1.index)] = [str(int(float(data1.loc[i]['user_id']))), str(int(float(data1.loc[i]['article_id']))), str(data1.loc[i]['content'])]
# df1.to_excel(path3, index=False)
# df2 = pd.DataFrame(columns=['user_id', 'content', 'ref_id'])
# data2 = pd.read_excel(path2)
# for i in range(len(data2)):
#     print(i)
#     if type(data2.loc[i]['content']) is str:
#         df2.loc[len(df2.index)] = [str(int(float(data2.loc[i]['user_id']))), str(data2.loc[i]['content']), str(int(float(data2.loc[i]['ref_id'])))]
# df2.to_excel(path4, index=False)

'''数据合并'''
path5 = r'E:\毕业论文\数据集\台湾选举\twitter3.xlsx'
# data1 = pd.read_excel(path3).astype(str)
# df1 = data1.drop_duplicates(subset=['user_id', 'ref_id'])
# data2 = pd.read_excel(path4).astype(str)
# df = pd.merge(df1, data2, on='ref_id', how='right').astype(str)
# df.to_excel(path5, index=False)

'''情感抽取'''
path6 = r'E:\毕业论文\数据集\台湾选举\tw_zf.csv'
path7 = r'E:\毕业论文\数据集\台湾选举\tw_st.csv'
# st = ['侯友宜', '趙少康', '柯文哲', '吳欣盈', '藍白合', '賴清德', '蕭美琴', '賴蕭配', '郭臺銘', '賴佩霞', '國民黨', '民進黨', '民衆黨']
# data = pd.read_excel(path5)
# df1 = pd.DataFrame(columns=['from', 'to', 'sign'])
# df2 = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(data)):
#     print(i)
#     text1 = data.loc[i]['content_x']
#     text2 = data.loc[i]['content_y']
#     for j in range(len(st)):
#         if st[j] in text1:
#             words1 = jieba.lcut(text1)
#             sentence1 = ' '.join(words1)
#             polarity1 = SnowNLP(sentence1).sentiments * 2 - 1
#             words2 = jieba.lcut(text2)
#             sentence2 = ' '.join(words2)
#             polarity2 = SnowNLP(sentence2).sentiments * 2 - 1
#             if (polarity1 > 0.5 or polarity1 < -0.5) and (polarity2 > 0.5 or polarity2 < -0.5):
#                 # df1.loc[len(df1.index)] = [str(int(float(data.loc[i]['user_id_x'])))+'\t',
#                 #                            str(int(float(data.loc[i]['user_id_y'])))+'\t',
#                 #                            float(polarity1) * float(polarity2)]
#                 df2.loc[len(df2.index)] = [str(j), str(int(float(data.loc[i]['user_id_x'])))+'\t', float(polarity1)]
#                 df2.loc[len(df2.index)] = [str(j), str(int(float(data.loc[i]['user_id_y'])))+'\t', float(polarity2)]
#             break
# # st1 = df1.groupby(['from', 'to'])['sign'].apply(lambda x: x.mean()).reset_index()
# # st1.to_csv(path6, index=False)
# st2 = df2.groupby(['from', 'to'])['sign'].apply(lambda x: x.mean()).reset_index()
# st2.to_csv(path7, index=False)

'''转发网络分割'''
# df = pd.read_csv(path6)
# mask = df['from'] > df['to']
# df.loc[mask, ['from', 'to']] = df.loc[mask, ['to', 'from']].values
# df = df[df['from'] != df['to']]
# df = df.groupby(['from', 'to'])['sign'].apply(lambda x: x.mean()).reset_index().astype(str)
# counter = Counter(pd.concat([df['from'], df['to']]))
# result_list = [item for item, count in counter.items() if count > 1]
# mask = df['from'].isin(result_list) & df['to'].isin(result_list)
# df1 = df[mask].reset_index(drop=True)
# positive_edges = []
# negative_edges = []
# for i in range(len(df1)):
#     if df1.loc[i]['sign'] > '0':
#         positive_edges.append(list(df1.loc[i]))
#     else:
#         negative_edges.append(list(df1.loc[i]))
# pos_count = len(positive_edges)
# neg_count = len(negative_edges)
# print(pos_count, neg_count)
# neg_num = int(neg_count/3)
# neg = [negative_edges[:neg_num], negative_edges[neg_num:2*neg_num], negative_edges[2*neg_num:]]
# pos = [positive_edges, positive_edges, positive_edges]
# for k in range(3):
#     print(k)
#     path8 = r'E:\毕业论文\数据集\台湾选举\tw_zf' + str(k+1) + '.csv'
#     path9 = r'E:\毕业论文\数据集\台湾选举\user' + str(k+1) + '.txt'
#     balanced_data = pos[k] + neg[k]
#     user = []
#     for i in balanced_data:
#         user.append(i[0])
#         user.append(i[1])
#     user = sorted(list(set(user)))
#     df2 = pd.DataFrame(balanced_data, columns=['from', 'to', 'sign'])
#     node_mapping = {old_id: new_id for new_id, old_id in enumerate(user)}
#     df2["from"] = df2["from"].map(node_mapping)
#     df2["to"] = df2["to"].map(node_mapping)
#     df2['sign'] = np.where(df2['sign'] > '0', 1, -1)
#     df2.to_csv(path8, index=False)
#     f3 = open(path9, 'w', encoding='utf-8')
#     for i in user:
#         f3.writelines([str(i), '\n'])
#     f3.close()

'''实体网络分割'''
# data = pd.read_csv(path7).astype(str)
# for k in range(3):
#     path8 = r'E:\毕业论文\数据集\台湾选举\tw_st' + str(k+1) + '.csv'
#     path9 = r'E:\毕业论文\数据集\台湾选举\user' + str(k+1) + '.txt'
#     with open(path9, "r") as f:
#         user = f.readlines()
#     user1 = []
#     for i in user:
#         user1.append(i[:-1])
#     mask = data['to'].isin(user1)
#     df1 = data[mask].reset_index(drop=True)
#     node_mapping = {old_id: new_id for new_id, old_id in enumerate(user1)}
#     df1["to"] = df1["to"].map(node_mapping)
#     st0 = sorted(list(set(list(df1['from']))))
#     node_mapping = {old_id: new_id for new_id, old_id in enumerate(st0, len(user1))}
#     df1["from"] = df1["from"].map(node_mapping)
#     df1.to_csv(path8)

'''转发网络生成'''
# with open(path8, "r") as f:
#     user = f.readlines()
# user1 = []
# for i in user:
#     user1.append(i[:-1])
# path9 = r'E:\毕业论文\数据集\台湾选举\tw_zf1.csv'
# data = pd.read_csv(path6).astype(str)
# df1 = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(data)):
#     if str(int(data.loc[i]['from'])) in user1 and str(int(data.loc[i]['to'])) in user1:
#         df1.loc[len(df1.index)] = list(data.loc[i])
# st = pd.read_csv(path7).astype(str)
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
# df2.to_csv(path9, index=False)
# st0 = list(set(list(st['from'])))
# df3 = pd.DataFrame(columns=['from', 'to', 'sign'])
# for i in range(len(st)):
#     if st.loc[i]['to'] in user2:
#         df3.loc[len(df3.index)] = [str(st0.index(st.loc[i]['from']) + len(user2)), str(user2.index(st.loc[i]['to'])), float(st.loc[i]['sign'])]
# path10 = r'E:\毕业论文\数据集\台湾选举\tw_st1.csv'
# df3.to_csv(path10, index=False)

'''数据清理'''
# path2 = r'E:\毕业论文\数据集\美国大选1225\mgdx_st.txt'
# path3 = r'E:\毕业论文\数据集\美国大选1225\mgdx_st_0.txt'
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
# f2 = open(path3, 'w', encoding='utf-8')
# for i in data1:
#     if i.split('\t')[1] in result_list:
#         f2.writelines([i.split('\t')[0], '\t', str(result_list.index(i.split('\t')[1])), '\t', i.split('\t')[2]])
# f2.close()
# f3 = open(path4, 'w', encoding='utf-8')
# for i in result_list:
#     f3.writelines([i, '\n'])
# f3.close()

'''转发网络提取'''
# path4 = r'E:\毕业论文\数据集\美国大选1225\user_0.txt'
# path5 = r'E:\毕业论文\数据集\美国大选1225\mgdx_zf_2.csv'
# path6 = r'E:\毕业论文\数据集\美国大选1225\user_2.txt'
# data = pd.read_csv(path, low_memory=False)
# with open(path4, "r") as f:
#     user = f.readlines()
# user1 = []
# for i in user:
#     user1.append(i[:-1])
# net = []
# # f1 = open(path5, 'w', encoding='utf-8')
# for i in range(len(data)):
#     if str(data.loc[i]['comment_count']) != 'nan':
#         if type(data.loc[i]['content']) is str and str(int(float(data.loc[i]['article_id']))) in user1:
#             if '[cmt]' in data.loc[i]['content']:
#                 text = data.loc[i]['content'].split('[cmt]')[0]
#             else:
#                 text = data.loc[i]['content']
#             blob = TextBlob(text)
#             polarity = blob.sentiment.polarity
#             net.append([str(int(float(data.loc[i]['root_article_id']))), str(int(float(data.loc[i]['article_id']))), str(polarity)])
# # f1.close()
# user = []
# for i in net:
#     if float(i[2]) != 0.0:
#         user.append(i[0])
#         user.append(i[1])
# user = list(set(user))
# df = pd.DataFrame(columns=['from', 'to', 'sign'])
# # f2 = open(path6, 'w', encoding='utf-8')
# for i in net:
#     if float(i[2]) > 0:
#         df.loc[len(df.index)] = [str(user.index(i[0])), str(user.index(i[1])), '1']
#         # f2.writelines([str(user.index(i.split('\t')[0])), '\t', str(user.index(i.split('\t')[1])), '\t', '1\n'])
#     elif float(i[2]) < 0:
#         df.loc[len(df.index)] = [str(user.index(i[0])), str(user.index(i[1])), '-1']
#         # f2.writelines([str(user.index(i.split('\t')[0])), '\t', str(user.index(i.split('\t')[1])), '\t', '-1\n'])
# # f2.close()
# for i in range(len(df)):
#     if int(df.loc[i]['from']) > int(df.loc[i]['to']):
#         tmp = int(df.loc[i]['from'])
#         df.loc[i]['from'] = int(df.loc[i]['to'])
#         df.loc[i]['to'] = tmp
# result = df.groupby(['from', 'to'])['sign'].apply(lambda x: x.max()).reset_index()
# result.to_csv(path5)
# f3 = open(path6, 'w', encoding='utf-8')
# for i in user:
#     f3.writelines([i, '\n'])
# f3.close()

'''实体网络补全'''
# path6 = r'E:\毕业论文\数据集\美国大选1225\user_1.txt'
# path7 = r'E:\毕业论文\数据集\美国大选1225\mgdx_st_3.csv'
# data = pd.read_csv(path, low_memory=False)
# with open(path6, "r") as f:
#     user = f.readlines()
# user1 = []
# for i in user:
#     user1.append(i[:-1])
# df = pd.DataFrame(columns=['source', 'target', 'sign'])
# for i in range(len(data)):
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
#                     df.loc[len(df.index)] = ['0', str(user1.index(str(int(float(data.loc[i]['article_id']))))), float(polarity)]
#
#             elif 'Trump' in text or 'trump' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = ['1', str(user1.index(str(int(float(data.loc[i]['article_id']))))), float(polarity)]
#
#             elif 'Harris' in text or 'harris' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = ['2', str(user1.index(str(int(float(data.loc[i]['article_id']))))), float(polarity)]
#
#             elif 'Vance' in text or 'vance' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = ['3', str(user1.index(str(int(float(data.loc[i]['article_id']))))), float(polarity)]
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
#                     df.loc[len(df.index)] = ['0', str(user1.index(str(int(float(data.loc[i]['root_article_id']))))), float(polarity)]
#
#             elif 'Trump' in text or 'trump' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = ['1', str(user1.index(str(int(float(data.loc[i]['root_article_id']))))), float(polarity)]
#
#             elif 'Harris' in text or 'harris' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = ['2', str(user1.index(str(int(float(data.loc[i]['root_article_id']))))), float(polarity)]
#
#             elif 'Vance' in text or 'vance' in text:
#                 blob = TextBlob(text)
#                 polarity = blob.sentiment.polarity
#                 if float(polarity) != 0.0:
#                     df.loc[len(df.index)] = ['3', str(user1.index(str(int(float(data.loc[i]['root_article_id']))))), float(polarity)]
# result = df.groupby(['source', 'target'])['sign'].apply(lambda x: x.mean()).reset_index()
# for i in range(len(user1)):
#     if str(i) not in list(result['target']):
#         print(i)
#         result.loc[len(result.index)] = [str(random.randint(0, 3)), str(i), random.random()*2-1]
# # df1 = pd.DataFrame(columns=['from', 'to', 'sign'])
# # for i in range(len(result)):
# #     if result.loc[i]['sign'] > 0:
# #         df1.loc[len(df1.index)] = [result.loc[i]['source'], result.loc[i]['target'], '1']
# #     elif result.loc[i]['sign'] < 0:
# #         df1.loc[len(df1.index)] = [result.loc[i]['source'], result.loc[i]['target'], '-1']
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

# path13 = r'E:\毕业论文\数据集\台湾选举\tw_zf00.csv'
# path10 = r'E:\毕业论文\数据集\台湾选举\tw_zf01.csv'
# data = pd.read_csv(path13)
# df = pd.DataFrame(columns=['from', 'to', 'sign'])
# x = 0
# for i in range(len(data)):
#     if data.loc[i]['sign'] == -1 and x < 2500:
#         df.loc[len(df.index)] = [data.loc[i]['from'], data.loc[i]['to'], data.loc[i]['sign']]
#         x += 1
#     elif data.loc[i]['sign'] == 1:
#         df.loc[len(df.index)] = [data.loc[i]['from'], data.loc[i]['to'], data.loc[i]['sign']]
# df.to_csv(path10, index=False)

'''词频统计'''
# st = ['侯友宜', '趙少康', '柯文哲', '吳欣盈', '藍白合', '賴清德', '蕭美琴', '賴蕭配', '郭臺銘', '賴佩霞', '國民黨', '民進黨', '民衆黨']
# data = pd.read_excel(path5)
# con = list(set(list(data['content_x']) + list(data['content_y'])))
# for i in st:
#     x = 0
#     for j in con:
#         if i in j:
#             x += 1
#     print(i, x)
word_counts = {'侯友宜': 20007, '赵少康': 9252, '柯文哲': 12671, '吴欣盈': 764, '赖清德': 30056, '萧美琴': 13068,
               '赖萧配': 4489, '郭台铭': 1034, '赖佩霞': 489, '国民党': 10729, '民进党': 14693, '民众党': 3041, '王建煊': 41, '苏焕智': 18}
wordcloud = WordCloud(font_path='C:/Windows/Fonts/simhei.ttf',
                      background_color='white',
                      max_words=200,
                      max_font_size=100,
                      scale=2)
wordcloud.generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 不显示坐标轴
plt.show()
