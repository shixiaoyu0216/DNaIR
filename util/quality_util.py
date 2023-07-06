import numpy as np
import pandas as pd
from tqdm import tqdm
from util.jsondict_util import save_dict


def item_quality_generate(dat_path, qua_path):
    df = pd.read_csv(dat_path, sep=',', names=['user_id', 'item_id', 'ratings', 'timestamp'])

    item_dup = df['item_id'].drop_duplicates().to_numpy()
    quality_dict = {}
    quality_list = []
    for each_idx in tqdm(range(len(item_dup)), total=len(item_dup)):
        item_frame = df[df['item_id'] == item_dup[each_idx]]
        ratings_list = item_frame['ratings'].tolist()
        ratings_list = [1 if i > 3 else 0 for i in ratings_list]
        q_ik = np.array(ratings_list)
        theta = (np.sum(q_ik) + 1) / (len(q_ik) + 2)
        quality_dict[int(item_dup[each_idx])] = theta
        quality_list.append(theta)
    quality_list.sort(reverse=True)
    save_dict(qua_path, quality_dict)
