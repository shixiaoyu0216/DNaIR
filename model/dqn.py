import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

nn.LayerNorm = LayerNorm

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        q_values = torch.tanh(self.mu(x))
        return q_values


class DQN(object):
    def __init__(self, n_states, n_actions,
                 memory_capacity, lr, epsilon, target_network_replace_freq, batch_size, gamma, tau, K):
        self.n_states = n_states
        self.n_actions = n_actions
        self.memory_capacity = memory_capacity
        self.lr = lr
        self.epsilon = epsilon
        self.replace_freq = target_network_replace_freq
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.eval_net = Net(self.n_states, self.n_actions, 256)
        self.target_net = Net(self.n_states, self.n_actions, 256)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        hard_update(self.target_net, self.eval_net)
        if (torch.cuda.is_available()):
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
            self.loss_func = self.loss_func.cuda()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.K = K
        self.memory = np.zeros((0, self.n_states * 2 + 2))

    def choose_action(self, state, env, I_sim_list):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0)
        if (torch.cuda.is_available()):
            state = state.cuda()
        actions_Q = self.eval_net.forward(state)
        temp_actions_Qvalue = actions_Q.cpu().detach().numpy()

        temp_actions_sim_Qvalue = []
        for item_id in I_sim_list:
            index = env.action_space.index(item_id)
            temp_actions_sim_Qvalue.append(temp_actions_Qvalue[0][index])
        actions_sim_Qvalue = torch.from_numpy(np.array(temp_actions_sim_Qvalue))
        actions_sim_Qvalue = torch.unsqueeze(actions_sim_Qvalue, 0)
        actions_sim_Qvalue_list = actions_sim_Qvalue.tolist()[0]

        rec_list = []
        while len(rec_list) < self.K:
            if np.random.uniform() < self.epsilon:
                if len(actions_sim_Qvalue_list) > 0:
                    action_index = actions_sim_Qvalue_list.index(max(actions_sim_Qvalue_list, default=0))
                    action = I_sim_list[action_index]
                    I_sim_list.remove(action)
                    actions_sim_Qvalue_list.remove(max(actions_sim_Qvalue_list))
                else:
                    action = np.random.randint(0, self.n_actions)
            else:
                action = np.random.randint(0, self.n_actions)
            if action not in env.mask_list:
                rec_list.append(action)
                env.mask_list.append(action)
        return rec_list

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        if len(self.memory) < self.memory_capacity:
            self.memory = np.append(self.memory, [transition], axis=0)
        else:
            index = self.memory_counter % self.memory_capacity
            self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_freq == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(len(self.memory), self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_state = Variable(torch.tensor(batch_memory[:, :self.n_states], dtype=torch.float32))
        batch_action = Variable(
            torch.tensor(batch_memory[:, self.n_states:self.n_states + 1].astype(int), dtype=torch.long))
        batch_reward = Variable(torch.tensor(batch_memory[:, self.n_states + 1:self.n_states + 2], dtype=torch.float32))
        batch_state_ = Variable(torch.tensor(batch_memory[:, -self.n_states:], dtype=torch.float32))

        if (torch.cuda.is_available()):
            batch_state = batch_state.cuda()
            batch_action = batch_action.cuda()
            batch_reward = batch_reward.cuda()
            batch_state_ = batch_state_.cuda()

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_state_).detach()
        q_target = batch_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        soft_update(self.target_net, self.eval_net, self.tau)
