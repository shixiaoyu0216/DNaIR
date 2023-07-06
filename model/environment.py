import math
import numpy as np


class Env():
    def __init__(self, user, observation_data, I,
                 item_sim_matrix, item_pop_dict, quality_dict, mask_list, K):
        self.observation = np.array(observation_data)
        self.n_observation = len(self.observation)
        self.action_space = I
        self.n_actions = len(self.action_space)
        self.user = user
        self.item_sim_matrix = item_sim_matrix
        self.item_pop_dict = item_pop_dict
        self.quality_dict = quality_dict
        self.mask_list = mask_list
        self.K = K

    def reset(self, observation):
        self.observation = observation
        self.n_observation = len(self.observation)
        return self.observation

    def step(self, action):
        done = False
        s = self.observation

        if s[-1] == action:
            self.item_sim_matrix[str(s[-1])][str(action)] = 0
            r = -1
        else:
            quality = self.quality_dict[str(action)]
            r_div = 0.4 * quality * 1 / math.log((self.item_pop_dict[str(action)] + 1.1), 10)
            r_acc = 0
            for i in range(self.n_observation):
                if str(s[-(i + 1)]) in self.item_sim_matrix.keys():
                    if str(action) in self.item_sim_matrix[str(s[-(i + 1)])].keys():
                        r_acc += (0.9 ** i) * self.item_sim_matrix[str(s[-(i + 1)])][str(action)]
            r = r_acc + r_div
        if r > 0:
            s_temp_ = np.append(s, action)
            observation_ = np.delete(s_temp_, 0, axis=0)
        else:
            observation_ = s
        s_ = observation_
        return s_, r, done
