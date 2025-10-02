path1 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\vote_net50.txt'
path2 = r'C:\Users\YHR\Desktop\YHR\硕士\符号网络\数据\议员数据\投票数据\vote_net50_weight.txt'


with open(path1, "r") as f:
    data = f.readlines()
dic = {}
for i in data:
    dic.setdefault(int(i.split('\t')[0]), [])
    dic[int(i.split('\t')[0])].append(int(i.split('\t')[2][:-1]))
dic0 = sorted(dic.items(), key=lambda item: item[0])
print(dic0)
# w1 = []
# w2 = []
# for i in range(50):
#     x = 0
#     y = 0
#     for j in dic0:
#         x += abs(j[1][i])
#         if j[1][i] > 0:
#             y += j[1][i]
#     w1.append(abs(0.5 - y / x))
#
# for k in range(50):
#     ph = 0
#     cs = -1
#     for v in range(50):
#         A = 0
#         B = 0
#         C = 0
#         D = 0
#         E = 0
#         for i in range(len(dic0) - 1):
#             for j in range(i + 1, len(dic0)):
#                 die = [dic0[i][1][k], dic0[j][1][k], dic0[j][1][v], dic0[i][1][v]]
#                 if die == [1, 1, 1, 1]:
#                     A += 1
#                 elif die == [-1, -1, -1, -1]:
#                     B += 1
#                 elif die.count(-1) == 2 and die.count(1) == 2:
#                     C += 1
#                 elif die.count(-1) == 1 and die.count(1) == 3:
#                     D += 1
#                 elif die.count(-1) == 3 and die.count(1) == 1:
#                     E += 1
#         SUM = A + B + C + D + E
#         if (D+E)/SUM < 0.3:
#             ph += (D+E)/SUM
#             cs += 1
#     if cs == 0:
#         w2.append(0.3)
#     else:
#         w2.append(ph/cs)
#
# f = open(path2, 'w', encoding='utf-8')
# for i in range(50):
#     f.writelines([str(i), '\t', str(w1[i]), '\t', str(w2[i]), '\n'])
# f.close()
