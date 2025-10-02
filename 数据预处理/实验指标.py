import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


'''链路预测'''
path1 = r'E:\毕业论文\对比方法\sigat\tw1.npy'
path2 = r'E:\毕业论文\数据集\台湾选举\tw_zf1.csv'
# path2 = r'E:\毕业论文\数据集\哇哈哈\Wahaha_zf1.csv'
emb = np.load(path1)
# emb = pd.read_csv(path1, header=None)
dataset = pd.read_csv(path2).values.tolist()
edges = {"positive_edges": [edge[0:2] for edge in dataset if edge[2] == 1],
         "negative_edges": [edge[0:2] for edge in dataset if edge[2] == -1], "ecount": len(dataset),
         "ncount": len(set([edge[0] for edge in dataset] + [edge[1] for edge in dataset]))}
test_y_edges = edges["positive_edges"] + edges["negative_edges"]
test_y_num = len(test_y_edges)*0.3
test_n_edges = []
for i in range(edges["ncount"] - 1):
    for j in range(i, edges["ncount"]):
        if np.random.rand() < 0.01:
            if [i, j] not in edges["positive_edges"] and [i, j] not in edges["negative_edges"]:
                test_n_edges.append([i, j])
            if len(test_n_edges) > test_y_num:
                break
        if len(test_n_edges) > test_y_num:
            break
edges1 = [edge[:]+[1] for edge in test_y_edges] + [edge[:]+[0] for edge in test_n_edges]

features = []
labels = []
for edge in edges1:
    node_i, node_j, label = edge
    feature = np.concatenate([emb[node_i], emb[node_j]])  # 拼接节点嵌入
    features.append(feature)
    labels.append(label)

features = np.array(features)
labels = np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]  # 预测概率
y_pred_p = [4*(p-0.5)*(p-0.5) for p in y_pred_prob]
y_pred = [1 if (p > 0.66 or p < 0.33) else 0 for p in y_pred_prob]
y_pred_1 = [1 if 0.5 else 0 for p in y_pred_prob]
auc_score = roc_auc_score(y_test, y_pred_p)
f1 = f1_score(y_test, y_pred)
print(f"AUC Score: {auc_score:.4f}")
print(f"F1 Score: {f1:.4f}")


'''符号预测'''
# emb = pd.read_csv(path1, header=None)
# dataset = pd.read_csv(path2).values.tolist()
# edges = {"positive_edges": [edge[0:2] for edge in dataset if edge[2] == 1],
#          "negative_edges": [edge[0:2] for edge in dataset if edge[2] == -1], "ecount": len(dataset),
#          "ncount": len(set([edge[0] for edge in dataset] + [edge[1] for edge in dataset]))}
# edges1 = [edge[:]+[1] for edge in edges["positive_edges"]] + [edge[:]+[0] for edge in edges["negative_edges"]]
# features = []
# labels = []
# for edge in edges1:
#     node_i, node_j, label = edge
#     feature = np.concatenate([emb.loc[node_i], emb.loc[node_j]])  # 拼接节点嵌入
#     features.append(feature)
#     labels.append(label)
#
# features = np.array(features)
# labels = np.array(labels)
# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# model = LogisticRegression()
# model.fit(X_train, y_train)
# y_pred_prob = model.predict_proba(X_test)[:, 1]
# y_pred = [1 if p > 0.5 else 0 for p in y_pred_prob]
# auc_score = roc_auc_score(y_test, y_pred_prob)
# f1 = f1_score(y_test, y_pred)
# print(f"AUC Score: {auc_score:.4f}")
# print(f"F1 Score: {f1:.4f}")
