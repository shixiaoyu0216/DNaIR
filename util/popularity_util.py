import pandas as pd
from tqdm import tqdm
from util.jsondict_util import save_dict


def item_popularity_generate(dat_path, pop_path):
    df = pd.read_csv(dat_path, sep=',', names=['user_id', 'item_id', 'ratings', 'timestamp'])

    train_dict = {}
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        train_dict.setdefault(int(row['user_id']), list())
        train_dict[int(row['user_id'])].append(int(row['item_id']))

    item_popularity = dict()
    for user, items in train_dict.items():
        for item in items:
            item_popularity.setdefault(item, 0)
            item_popularity[item] += 1

    min_value = min(item_popularity.values())
    max_value = max(item_popularity.values())
    for key in item_popularity:
        normalized_value = (item_popularity[key] - min_value) / (max_value - min_value)
        item_popularity[key] = normalized_value

    save_dict(pop_path, item_popularity)
