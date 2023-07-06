import itertools

import numpy as np


def ils_metric(rec_list, item_sim_matrix):
    sim_temp = 0
    for i in range(0, len(rec_list)):
        for j in range(i + 1, len(rec_list)):
            if str(rec_list[j]) in item_sim_matrix[str(rec_list[i])]:
                sim_temp += item_sim_matrix[str(rec_list[i])][str(rec_list[j])]
    return 1 - (sim_temp / (len(rec_list) * (len(rec_list) - 1)))


def ndcg_metric(rec_list, test_dict):
    ndcg = 0
    for key, topn_set in rec_list.items():
        test_set = test_dict.get(key)
        dsct_list = [1 / np.log2(i + 1) for i in range(1, len(topn_set) + 1)]
        z_k = sum(dsct_list)
        if test_set is not None:
            mask = [0 if i not in test_set else 1 for i in topn_set]
            ndcg += sum(np.multiply(dsct_list, mask)) / z_k
    ndcg = ndcg / len(rec_list.items())
    return ndcg


def novelty_metric(rec_list, pop_dict):
    pop_sum = []
    for item in rec_list:
        if str(item) in pop_dict.keys():
            pop_sum.append(pop_dict[str(item)])
    return np.mean(pop_sum)


def interdiv_metric(interdiv_list):
    interdiv_result = 0
    interdiv_comb_list = list(itertools.combinations(interdiv_list, 2))
    for each_comb in interdiv_comb_list:
        temp_comb = []
        temp_comb.extend(each_comb[0])
        temp_comb.extend(each_comb[1])
        interdiv_result += len(list(set(each_comb[0]) & set(each_comb[1]))) / (len(each_comb[0]))
    return 2 * interdiv_result / len(interdiv_comb_list)
