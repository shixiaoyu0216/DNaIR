import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait
from model import environment, dqn
from util.metrics_util import ndcg_metric, novelty_metric, ils_metric, interdiv_metric

user_num = 0
precision, ndcg, novelty, coverage, ils, interdiv = [], [], [], [], [], []


def setInteraction(env, agent, ep_user, train_df, obswindow):
    user_df = train_df[train_df['user_id'] == ep_user]
    state_list = []
    for obs in user_df['item_id'].rolling(obswindow):
        if len(obs) != obswindow:
            continue
        state_list.append(list(obs))
    interaction_num = 0
    for s_idx in range(len(state_list) - 1):
        s = np.array(env.reset(state_list[s_idx]))
        a = int(state_list[s_idx + 1][0])
        s_, r, done = env.step(a)
        agent.store_transition(s, a, r, s_)
        interaction_num += 1
    return interaction_num


def recommend_offpolicy(env, agent, last_obs):
    state = np.array(last_obs)
    s = env.reset(state)

    item_sim_dict_1 = env.item_sim_matrix[str(s[-1])]
    item_sim_dict_2 = {}
    for each_item in item_sim_dict_1.keys():
        if int(each_item) not in env.mask_list:
            item_sim_dict_2[int(each_item)] = item_sim_dict_1[each_item]
    sorted_I = sorted(item_sim_dict_2.items(), key=lambda x: x[1], reverse=True)
    index = env.K
    I_sim, I_div = sorted_I[:index], sorted_I[index:]
    I_sim_list = [list(i)[0] for i in I_sim]

    return agent.choose_action(s, env, I_sim_list)


def trainAgent(agent, step_max):
    step = 0
    while step < step_max:
        agent.learn()
        step += 1


def recommender(ep_user, train_df, test_df, train_dict,
                item_sim_dict, item_quality_dict, item_pop_dict,
                max_item_id, mask_list, args):
    last_obs = train_dict[ep_user][-args.obswindow:]
    mask_list.extend(train_dict[ep_user][:-1])
    env = environment.Env(ep_user, train_dict[ep_user][-args.obswindow:], list(range(max_item_id)),
                          item_sim_dict, item_pop_dict, item_quality_dict, mask_list, args.topk)
    agent = dqn.DQN(args.obswindow, env.n_actions,
                    args.memory, args.lr, args.epsilon,
                    args.replace_freq, args.batch, args.gamma, args.tau, args.topk)

    interaction_num = setInteraction(env, agent, ep_user, train_df, args.obswindow)
    if interaction_num <= 20:
        return
    else:
        global user_num
        user_num += 1

    trainAgent(agent, args.step_max)
    rec_list = recommend_offpolicy(env, agent, last_obs)
    test_set = test_df.loc[test_df['user_id'] == ep_user, 'item_id'].tolist()

    global precision, ndcg, novelty, coverage, ils, interdiv
    precision.append(len(set(rec_list) & set(test_set)) / (len(rec_list)))
    ndcg.append(ndcg_metric({ep_user: rec_list}, {ep_user: test_set}))
    novelty.append(novelty_metric(rec_list, env.item_pop_dict))
    coverage.extend(rec_list)
    ils.append(ils_metric(rec_list, env.item_sim_matrix))
    interdiv.append(rec_list)


def train_dqn(train_df, test_df,
              item_sim_dict, item_quality_dict, item_pop_dict,
              max_item_id, item_list, mask_list, args):
    train_dict = {}
    for index, row in train_df.iterrows():
        train_dict.setdefault(int(row['user_id']), list())
        train_dict[int(row['user_id'])].append(int(row['item_id']))

    futures = []
    executor = ThreadPoolExecutor(max_workers=args.j)
    train_episodes = random.sample(list(train_dict.keys()), args.episode_max)
    for ep_user in train_episodes:
        future = executor.submit(recommender,
                                 ep_user, train_df, test_df, train_dict,
                                 item_sim_dict, item_quality_dict, item_pop_dict,
                                 max_item_id, mask_list, args)
        futures.append(future)
    wait(futures)

    print("Precision: ", np.mean(precision))
    print("NDCG: ", np.mean(ndcg))
    print("Novelty: ", 1 - np.mean(novelty))
    print("ILS: ", np.mean(ils))


if __name__ == "__main__":
    pass
