import pandas as pd
import json


# 电影
# path = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\电影\ml-latest-small\ratings.csv'
# path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\电影\movie_net.txt'
# path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\电影\movie_nodes.txt'
# df = pd.read_csv(path)
# movie = list(set(list(df['movieId'])))
# movie.sort()
# score = []
# for j in movie:
#     x = 0
#     y = 0
#     filtered_df = df[df['movieId'] == j]['rating']
#     score.append(sum(list(filtered_df)) / len(list(filtered_df)))
#     print(j)
# f1 = open(path1, 'w', encoding='utf-8')
# for i in range(len(df)):
#     f1.writelines([str(df.iloc[i, 0]), '\t', str(movie.index(df.iloc[i, 1])), '\t', str((df.iloc[i, 2] - 3) / 2), '\n'])
# f1.close()
# f2 = open(path2, 'w', encoding='utf-8')
# for i in range(len(movie)):
#     f2.writelines([str(movie[i]), '\t', str(i), '\t', str(score[i]), '\n'])
# f2.close()

# 购物
# path = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Amazon\Amazon.csv'
# path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Amazon\Amazon_net.txt'
# path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Amazon\Amazon_goods.txt'
# path3 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Amazon\Amazon_users.txt'
# df = pd.read_csv(path, header=None)
# user = list(set(list(df[0])))
# thing = list(set(list(df[1])))
# print(len(user), len(thing))
# score = []
# for j in thing:
#     x = 0
#     y = 0
#     filtered_df = df[df[1] == j][2]
#     score.append(sum(list(filtered_df)) / len(list(filtered_df)))
#     print(j)
# f1 = open(path1, 'w', encoding='utf-8')
# for i in range(len(df)):
#     f1.writelines([str(user.index(df.iloc[i, 0])), '\t', str(thing.index(df.iloc[i, 1])), '\t', str((df.iloc[i, 2] - 3) / 2), '\n'])
# f1.close()
# f2 = open(path2, 'w', encoding='utf-8')
# for i in range(len(thing)):
#     f2.writelines([str(thing[i]), '\t', str(i), '\t', str(score[i]), '\n'])
# f2.close()
# f3 = open(path3, 'w', encoding='utf-8')
# for i in range(len(user)):
#     f3.writelines([str(user[i]), '\t', str(i), '\n'])
# f3.close()

# 豆瓣电影
# path = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\豆瓣电影\豆瓣电影原始数据集\movies.csv'
# path0 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\豆瓣电影\豆瓣电影原始数据集\ratings.csv'
# path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Amazon\Amazon_net.txt'
# path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Amazon\Amazon_goods.txt'
# path3 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Amazon\Amazon_users.txt'
# df = pd.read_csv(path)
# df0 = df[df['DOUBAN_SCORE'] != 0]['MOVIE_ID']
# movie = list(df0)
# df1 = pd.read_csv(path0)
# df2 = pd.merge(df1, df0, how='inner', on='MOVIE_ID')
# print(len(df2))
# user = list(set(list(df[0])))
# thing = list(set(list(df[1])))
# print(len(user), len(thing))
# score = []
# for j in thing:
#     x = 0
#     y = 0
#     filtered_df = df[df[1] == j][2]
#     score.append(sum(list(filtered_df)) / len(list(filtered_df)))
#     print(j)
# f1 = open(path1, 'w', encoding='utf-8')
# for i in range(len(df)):
#     f1.writelines([str(user.index(df.iloc[i, 0])), '\t', str(thing.index(df.iloc[i, 1])), '\t', str((df.iloc[i, 2] - 3) / 2), '\n'])
# f1.close()
# f2 = open(path2, 'w', encoding='utf-8')
# for i in range(len(thing)):
#     f2.writelines([str(thing[i]), '\t', str(i), '\t', str(score[i]), '\n'])
# f2.close()
# f3 = open(path3, 'w', encoding='utf-8')
# for i in range(len(user)):
#     f3.writelines([str(user[i]), '\t', str(i), '\n'])
# f3.close()

# amazon
path = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\archive\1429_1.csv'
path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\archive0\1429_net.txt'
path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\archive0\1429_goods.txt'
path3 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\archive0\1429_users.txt'
df = pd.read_csv(path)
df1 = df.dropna(subset=['reviews.rating', 'asins', 'reviews.username'])
user0 = list(df1['reviews.username'])
user1 = list(set(list(df1['reviews.username'])))
user = []
for i in user1:
    if user0.count(i) > 1:
        user.append(i)
df2 = df1[df1['reviews.username'].isin(user)]
thing = list(set(list(df2['asins'])))
print(len(user), len(thing))
score = []
for j in thing:
    filtered_df = df2[df2['asins'] == j]['reviews.rating']
    score.append(sum(list(filtered_df)) / len(list(filtered_df)))
    print(j)
f1 = open(path1, 'w', encoding='utf-8')
for i in range(len(df2)):
    if df2.iloc[i, 20] in user:
        f1.writelines([str(user.index(df2.iloc[i, 20])), '\t', str(thing.index(df2.iloc[i, 2])), '\t', str((df2.iloc[i, 14] - 3) / 2), '\n'])
f1.close()
f2 = open(path2, 'w', encoding='utf-8')
for i in range(len(thing)):
    f2.writelines([str(thing[i]), '\t', str(i), '\t', str(score[i]), '\n'])
f2.close()
f3 = open(path3, 'w', encoding='utf-8')
for i in range(len(user)):
    f3.writelines([str(user[i]), '\t', str(i), '\n'])
f3.close()

# path = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Magazine5\Magazine.json'
# path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Magazine5\Magazine_net.txt'
# path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Magazine5\Magazine_goods.txt'
# path3 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\购物\Magazine5\Magazine_users.txt'
# file = open(path, 'r', encoding='utf-8')
# data = []
# for line in file.readlines():
#     dic = json.loads(line)
#     data.append(dic)
# df = pd.DataFrame(columns=['id', 'asin', 'rating'])
# for i in data:
#     df.loc[len(df.index)] = [i['reviewerID'], i['asin'], i['overall']]
# user = list(set(list(df['id'])))
# thing = list(set(list(df['asin'])))
# print(len(user), len(thing))
# score = []
# for j in thing:
#     filtered_df = df[df['asin'] == j]['rating']
#     score.append(sum(list(filtered_df)) / len(list(filtered_df)))
#     print(j)
# f1 = open(path1, 'w', encoding='utf-8')
# for i in range(len(df)):
#     f1.writelines([str(user.index(df.iloc[i, 0])), '\t', str(thing.index(df.iloc[i, 1])), '\t', str((df.iloc[i, 2] - 3) / 2), '\n'])
# f1.close()
# f2 = open(path2, 'w', encoding='utf-8')
# for i in range(len(thing)):
#     f2.writelines([str(thing[i]), '\t', str(i), '\t', str(score[i]), '\n'])
# f2.close()
# f3 = open(path3, 'w', encoding='utf-8')
# for i in range(len(user)):
#     f3.writelines([str(user[i]), '\t', str(i), '\n'])
# f3.close()
