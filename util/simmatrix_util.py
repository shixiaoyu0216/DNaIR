import math
import pandas as pd
from tqdm import tqdm
from util.jsondict_util import save_dict


def sim_matrix_generate(dat_path, mat_path):
    df = pd.read_csv(dat_path, sep=',', names=['user_id', 'item_id', 'ratings', 'timestamp'])

    train_dict = {}
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        train_dict.setdefault(int(row['user_id']), list())
        train_dict[int(row['user_id'])].append(int(row['item_id']))

    item_sim_matrix = {}
    item_viewnum_dict = {}
    for user in tqdm(train_dict, total=len(train_dict)):
        items = train_dict[user]
        for i in range(len(items)):
            item_sim_matrix.setdefault(items[i], {})
            for j in range(len(items)):
                if j != i:
                    item_sim_matrix[items[i]].setdefault(items[j], 0)
                    item_sim_matrix[items[i]][items[j]] += 1
            item_viewnum_dict.setdefault(items[i], 0)
            item_viewnum_dict[items[i]] += 1

    for u in tqdm(item_sim_matrix.keys(), total=len(item_sim_matrix.keys())):
        for v in item_sim_matrix[u].keys():
            if item_sim_matrix[u][v] == 0:
                item_sim_matrix[u][v] = 0
            else:
                item_sim_matrix[u][v] /= math.sqrt(item_viewnum_dict[u] * item_viewnum_dict[v])

    save_dict(mat_path, item_sim_matrix)
