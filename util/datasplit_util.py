import pandas as pd
from tqdm import tqdm


def data_split(dat_path, train_path, valid_path, test_path):
    df = pd.read_csv(dat_path, sep=',', names=['user_id', 'item_id', 'ratings', 'timestamp'])

    grouped = df.groupby('user_id')

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    train_set = pd.DataFrame(columns=df.columns)
    val_set = pd.DataFrame(columns=df.columns)
    test_set = pd.DataFrame(columns=df.columns)

    for name, group in tqdm(grouped, total=len(grouped)):
        train_size = int(len(group) * train_ratio)
        val_size = int(len(group) * val_ratio)

        val = group.iloc[: val_size]
        train = group.iloc[val_size: train_size + val_size + 1]
        test = group.iloc[train_size + val_size:]

        train_set = pd.concat([train_set, train])
        val_set = pd.concat([val_set, val])
        test_set = pd.concat([test_set, test])

    train_set.to_csv(train_path, index=None, header=None)
    val_set.to_csv(valid_path, index=None, header=None)
    test_set.to_csv(test_path, index=None, header=None)
