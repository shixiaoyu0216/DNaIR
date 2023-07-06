import argparse
import os
import pandas as pd

from train import train_dqn
from util.datasplit_util import data_split
from util.jsondict_util import load_dict
from util.popularity_util import item_popularity_generate
from util.quality_util import item_quality_generate
from util.simmatrix_util import sim_matrix_generate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='kuairec')
    parser.add_argument('--obswindow', type=int, default=10)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--memory', type=int, default=20000)
    parser.add_argument('--replace_freq', type=int, default=99)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epsilon', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.90)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--episode_max', type=int, default=10)
    parser.add_argument('--step_max', type=int, default=100)
    parser.add_argument('--j', type=int, default=16)
    args = parser.parse_args()

    train_df = None
    test_df = None
    item_sim_dict = None
    item_quality_dict = None
    item_pop_dict = None
    max_item_id = None
    item_list = None
    mask_list = None

    dataset = args.dataset
    dat_path = './dataset/' + dataset + '/' + dataset + '.dat'
    if os.path.exists(dat_path):
        df = pd.read_csv(dat_path, sep=',',
                         names=['user_id', 'item_id', 'ratings', 'timestamp'])
        max_item_id = df['item_id'].max()
        item_list = df['item_id'].tolist()
        mask_list = list(set(list(range(max_item_id))) - set(item_list))
        mat_path = './dataset/' + dataset + '/' + dataset + '.mat'
        if os.path.exists(mat_path):
            item_sim_dict = load_dict(mat_path)
        else:
            sim_matrix_generate(dat_path, mat_path)
            item_sim_dict = load_dict(mat_path)
        qua_path = './dataset/' + dataset + '/' + dataset + '.qua'
        if os.path.exists(qua_path):
            item_quality_dict = load_dict(qua_path)
        else:
            item_quality_generate(dat_path, qua_path)
            item_quality_dict = load_dict(qua_path)
        pop_path = './dataset/' + dataset + '/' + dataset + '.pop'
        if os.path.exists(pop_path):
            item_pop_dict = load_dict(pop_path)
        else:
            item_popularity_generate(dat_path, pop_path)
            item_pop_dict = load_dict(pop_path)
        train_path = './dataset/' + dataset + '/' + dataset + '.train'
        valid_path = './dataset/' + dataset + '/' + dataset + '.valid'
        test_path = './dataset/' + dataset + '/' + dataset + '.test'
        if (os.path.exists(train_path)) \
                & (os.path.exists(valid_path)) \
                & (os.path.exists(test_path)):
            train_df = pd.read_csv(train_path, sep=',',
                                   names=['user_id', 'item_id', 'ratings', 'timestamp'])
            valid_df = pd.read_csv(valid_path, sep=',',
                                   names=['user_id', 'item_id', 'ratings', 'timestamp'])
            test_df = pd.read_csv(test_path, sep=',',
                                  names=['user_id', 'item_id', 'ratings', 'timestamp'])
        else:
            data_split(dat_path, train_path, valid_path, test_path)
            train_df = pd.read_csv(train_path, sep=',',
                                   names=['user_id', 'item_id', 'ratings', 'timestamp'])
            valid_df = pd.read_csv(valid_path, sep=',',
                                   names=['user_id', 'item_id', 'ratings', 'timestamp'])
            test_df = pd.read_csv(test_path, sep=',',
                                  names=['user_id', 'item_id', 'ratings', 'timestamp'])
    else:
        print("Please check if the dataset file exists!")

    train_dqn(train_df, test_df,
              item_sim_dict, item_quality_dict, item_pop_dict,
              max_item_id, item_list, mask_list, args)
