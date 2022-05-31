import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

import random
import numpy as np
import collections



class Policy(nn.Module):
    def __init__(self, input_shape, out_c, out_d):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_shape, 1024)  # Better result with slightly wider networks.
        self.fc2 = nn.Linear(1024, 1024)  # Better result with slightly wider networks.
        self.fc3 = nn.Linear(1024, 1024)  # Better result with slightly wider networks.
        self.fc4 = nn.Linear(1024, 1024)  # Better result with slightly wider networks.
        self.fc5 = nn.Linear(1024, 1024)  # Better result with slightly wider networks.
        self.mean = nn.Linear(1024, out_c)
        self.logstd = nn.Linear(1024, out_c)
        self.pi_d = nn.Linear(1024, out_d)
        self.weights_init = "orthogonal"
        self.bias_init = "zeros"

        # ALGO LOGIC: initialize agent here:
        self.LOG_STD_MAX = 0.0
        self.LOG_STD_MIN = -3.0

        self.apply(self.layer_init)

    def layer_init(self, layer, weight_gain=1, bias_const=0):
        if isinstance(layer, nn.Linear):
            if self.weights_init == "xavier":
                torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
            elif self.weights_init == "orthogonal":
                torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
            if self.bias_init == "zeros":
                torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, x, device):
        x = torch.Tensor(x).to(device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        mean = torch.tanh(self.mean(x))
        log_std = torch.tanh(self.logstd(x))
        pi_d = self.pi_d(x)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std, pi_d

    def get_action(self, x, device):
        mean, log_std, pi_d = self.forward(x, device)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action_c = torch.tanh(x_t)
        log_prob_c = normal.log_prob(x_t)
        log_prob_c -= torch.log(1.0 - action_c.pow(2) + 1e-8)

        dist = Categorical(logits=pi_d)
        action_d = dist.sample()
        prob_d = dist.probs
        log_prob_d = torch.log(prob_d + 1e-8)

        return action_c, action_d, log_prob_c, log_prob_d, prob_d

    def to(self, device):
        return super(Policy, self).to(device)



class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, out_c, out_d):
        super(SoftQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_shape + out_c, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, out_d)
        self.weights_init = "orthogonal"
        self.bias_init = "zeros"

        self.apply(self.layer_init)

    def layer_init(self, layer, weight_gain=1, bias_const=0):
        if isinstance(layer, nn.Linear):
            if self.weights_init == "xavier":
                torch.nn.init.xavier_uniform_(layer.weight, gain=weight_gain)
            elif self.weights_init == "orthogonal":
                torch.nn.init.orthogonal_(layer.weight, gain=weight_gain)
            if self.bias_init == "zeros":
                torch.nn.init.constant_(layer.bias, bias_const)

    def forward(self, x, a, device):
        x = torch.Tensor(x).to(device)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x



# modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py#
class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r)
            s_prime_lst.append(s_prime)
            done_mask_lst.append(done_mask)

        return np.array(s_lst), np.array(a_lst), \
               np.array(r_lst), np.array(s_prime_lst), \
               np.array(done_mask_lst)