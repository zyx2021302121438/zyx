from utils import setup_features
from param_parser import parameter_parser
from utils import tab_printer, read_graph
import numpy as np
import pandas as pd
from sgcn import SignedGCNTrainer
from param_parser import parameter_parser
from utils import tab_printer, read_graph, score_printer, save_logs, read_st_graph, train_test_split


def score_link_model(edges, t_positive_edges, t_negative_edges):
    test_y_edges = t_positive_edges + t_negative_edges
    test_y_num = len(test_y_edges)
    test_n_edges = []
    for i in range(edges["ncount"] - 1):
        for j in range(i, edges["ncount"]):
            # if np.random.rand() < 0.005:
            if [i, j] not in edges["positive_edges"] and [i, j] not in edges["negative_edges"]:
                test_n_edges.append([i, j])
            if len(test_n_edges) > test_y_num:
                break
        if len(test_n_edges) > test_y_num:
            break
    print(test_y_num, len(test_n_edges), test_n_edges)


# path = r'E:\SignNetwork\WBGCN-master\X.csv'
args = parameter_parser()
edges1 = read_graph(args)
positive_edges, test_positive_edges = train_test_split(edges1["positive_edges"], test_size=0.1)
negative_edges, test_negative_edges = train_test_split(edges1["negative_edges"], test_size=0.1)
score_link_model(edges1, test_positive_edges, test_negative_edges)
# df = pd.DataFrame(X)
# df.to_csv(path, index=False)
