
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, num_states, num_container, hidden1=256, hidden2=256, hidden3=256, hidden4=256, hidden5=256, hidden6=256, init_w=3e-3):
        super(Actor, self).__init__()

        # container deploy order
        # mean
        self.container_alpha_fc1 = nn.Linear(num_states, hidden1)
        self.container_alpha_fc2 = nn.Linear(hidden1, hidden2)
        self.container_alpha_fc3 = nn.Linear(hidden2, hidden3)
        self.container_alpha_fc4 = nn.Linear(hidden3, hidden4)
        self.container_alpha_fc5 = nn.Linear(hidden4, hidden5)
        self.container_alpha_fc6 = nn.Linear(hidden5, hidden6)
        self.container_alpha_fc7 = nn.Linear(hidden6, num_container)
        # std
        self.container_beta_fc1 = nn.Linear(num_states, hidden1)
        self.container_beta_fc2 = nn.Linear(hidden1, hidden2)
        self.container_beta_fc3 = nn.Linear(hidden2, hidden3)
        self.container_beta_fc4 = nn.Linear(hidden3, hidden4)
        self.container_beta_fc5 = nn.Linear(hidden4, hidden5)
        self.container_beta_fc6 = nn.Linear(hidden5, hidden6)
        self.container_beta_fc7 = nn.Linear(hidden6, num_container)

        # server preference
        # mean
        self.server_alpha_fc1 = nn.Linear(num_states+num_container+num_container, hidden1)
        self.server_alpha_fc2 = nn.Linear(hidden1, hidden2)
        self.server_alpha_fc3 = nn.Linear(hidden2, hidden3)
        self.server_alpha_fc4 = nn.Linear(hidden3, hidden4)
        self.server_alpha_fc5 = nn.Linear(hidden4, hidden5)
        self.server_alpha_fc6 = nn.Linear(hidden5, hidden6)
        self.server_alpha_fc7 = nn.Linear(hidden6, num_container)
        # std
        self.server_beta_fc1 = nn.Linear(num_states+num_container+num_container, hidden1)
        self.server_beta_fc2 = nn.Linear(hidden1, hidden2)
        self.server_beta_fc3 = nn.Linear(hidden2, hidden3)
        self.server_beta_fc4 = nn.Linear(hidden3, hidden4)
        self.server_beta_fc5 = nn.Linear(hidden4, hidden5)
        self.server_beta_fc6 = nn.Linear(hidden5, hidden6)
        self.server_beta_fc7 = nn.Linear(hidden6, num_container)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        # container deploy order
        # mean
        nn.init.orthogonal_(self.container_alpha_fc1.weight.data)
        self.container_alpha_fc2.weight.data = fanin_init(self.container_alpha_fc2.weight.data.size())
        self.container_alpha_fc3.weight.data = fanin_init(self.container_alpha_fc3.weight.data.size())
        self.container_alpha_fc4.weight.data = fanin_init(self.container_alpha_fc4.weight.data.size())
        self.container_alpha_fc5.weight.data = fanin_init(self.container_alpha_fc5.weight.data.size())
        self.container_alpha_fc6.weight.data = fanin_init(self.container_alpha_fc6.weight.data.size())
        self.container_alpha_fc7.weight.data.uniform_(-init_w, init_w)
        # std
        nn.init.orthogonal_(self.container_beta_fc1.weight.data)
        self.container_beta_fc2.weight.data = fanin_init(self.container_beta_fc2.weight.data.size())
        self.container_beta_fc3.weight.data = fanin_init(self.container_beta_fc3.weight.data.size())
        self.container_beta_fc4.weight.data = fanin_init(self.container_beta_fc4.weight.data.size())
        self.container_beta_fc5.weight.data = fanin_init(self.container_beta_fc5.weight.data.size())
        self.container_beta_fc6.weight.data = fanin_init(self.container_beta_fc6.weight.data.size())
        self.container_beta_fc7.weight.data.uniform_(-init_w, init_w)

        # server preference
        # mean
        nn.init.orthogonal_(self.server_alpha_fc1.weight.data)
        self.server_alpha_fc2.weight.data = fanin_init(self.server_alpha_fc2.weight.data.size())
        self.server_alpha_fc3.weight.data = fanin_init(self.server_alpha_fc3.weight.data.size())
        self.server_alpha_fc4.weight.data = fanin_init(self.server_alpha_fc4.weight.data.size())
        self.server_alpha_fc5.weight.data = fanin_init(self.server_alpha_fc5.weight.data.size())
        self.server_alpha_fc6.weight.data = fanin_init(self.server_alpha_fc6.weight.data.size())
        self.server_alpha_fc7.weight.data.uniform_(-init_w, init_w)
        # std
        nn.init.orthogonal_(self.server_beta_fc1.weight.data)
        self.server_beta_fc2.weight.data = fanin_init(self.server_beta_fc2.weight.data.size())
        self.server_beta_fc3.weight.data = fanin_init(self.server_beta_fc3.weight.data.size())
        self.server_beta_fc4.weight.data = fanin_init(self.server_beta_fc4.weight.data.size())
        self.server_beta_fc5.weight.data = fanin_init(self.server_beta_fc5.weight.data.size())
        self.server_beta_fc6.weight.data = fanin_init(self.server_beta_fc6.weight.data.size())
        self.server_beta_fc7.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        # container deploy order
        # mean
        container_alpha = self.container_alpha_fc1(state)
        container_alpha = self.relu(container_alpha)
        container_alpha = self.container_alpha_fc2(container_alpha)
        container_alpha = self.relu(container_alpha)
        container_alpha = self.container_alpha_fc3(container_alpha)
        container_alpha = self.relu(container_alpha)
        container_alpha = self.container_alpha_fc4(container_alpha)
        container_alpha = self.relu(container_alpha)
        container_alpha = self.container_alpha_fc5(container_alpha)
        container_alpha = self.relu(container_alpha)
        container_alpha = self.container_alpha_fc6(container_alpha)
        container_alpha = self.relu(container_alpha)
        container_alpha = self.container_alpha_fc7(container_alpha)
        container_alpha = self.softplus(container_alpha) + 1
        # std
        container_beta = self.container_beta_fc1(state)
        container_beta = self.relu(container_beta)
        container_beta = self.container_beta_fc2(container_beta)
        container_beta = self.relu(container_beta)
        container_beta = self.container_beta_fc3(container_beta)
        container_beta = self.relu(container_beta)
        container_beta = self.container_beta_fc4(container_beta)
        container_beta = self.relu(container_beta)
        container_beta = self.container_beta_fc5(container_beta)
        container_beta = self.relu(container_beta)
        container_beta = self.container_beta_fc6(container_beta)
        container_beta = self.relu(container_beta)
        container_beta = self.container_beta_fc7(container_beta)
        container_beta = self.softplus(container_beta) + 1

        # server preference
        # mean
        server_alpha = self.server_alpha_fc1(torch.cat([state, container_alpha, container_beta], -1))
        server_alpha = self.relu(server_alpha)
        server_alpha = self.server_alpha_fc2(server_alpha)
        server_alpha = self.relu(server_alpha)
        server_alpha = self.server_alpha_fc3(server_alpha)
        server_alpha = self.relu(server_alpha)
        server_alpha = self.server_alpha_fc4(server_alpha)
        server_alpha = self.relu(server_alpha)
        server_alpha = self.server_alpha_fc5(server_alpha)
        server_alpha = self.relu(server_alpha)
        server_alpha = self.server_alpha_fc6(server_alpha)
        server_alpha = self.relu(server_alpha)
        server_alpha = self.server_alpha_fc7(server_alpha)
        server_alpha = self.softplus(server_alpha) + 1
        # std
        server_beta = self.server_beta_fc1(torch.cat([state, container_alpha, container_beta], -1))
        server_beta = self.relu(server_beta)
        server_beta = self.server_beta_fc2(server_beta)
        server_beta = self.relu(server_beta)
        server_beta = self.server_beta_fc3(server_beta)
        server_beta = self.relu(server_beta)
        server_beta = self.server_beta_fc4(server_beta)
        server_beta = self.relu(server_beta)
        server_beta = self.server_beta_fc5(server_beta)
        server_beta = self.relu(server_beta)
        server_beta = self.server_beta_fc6(server_beta)
        server_beta = self.relu(server_beta)
        server_beta = self.server_beta_fc7(server_beta)
        server_beta = self.softplus(server_beta) + 1

        return container_alpha, container_beta, server_alpha, server_beta

class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden1=256, hidden2=256, hidden3=256, hidden4=256, hidden5=256, hidden6=256, init_w=3e-3):
        super(Critic, self).__init__()
        # Q1
        self.q1_fc1 = nn.Linear(num_states + num_actions, hidden1)
        self.q1_fc2 = nn.Linear(hidden1, hidden2)
        self.q1_fc3 = nn.Linear(hidden2, hidden3)
        self.q1_fc4 = nn.Linear(hidden3, hidden4)
        self.q1_fc5 = nn.Linear(hidden4, hidden5)
        self.q1_fc6 = nn.Linear(hidden5, hidden6)
        self.q1_fc7 = nn.Linear(hidden6, 1)
        # Q2
        self.q2_fc1 = nn.Linear(num_states + num_actions, hidden1)
        self.q2_fc2 = nn.Linear(hidden1, hidden2)
        self.q2_fc3 = nn.Linear(hidden2, hidden3)
        self.q2_fc4 = nn.Linear(hidden3, hidden4)
        self.q2_fc5 = nn.Linear(hidden4, hidden5)
        self.q2_fc6 = nn.Linear(hidden5, hidden6)
        self.q2_fc7 = nn.Linear(hidden6, 1)

        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        # Q1
        nn.init.orthogonal_(self.q1_fc1.weight.data)
        self.q1_fc2.weight.data = fanin_init(self.q1_fc2.weight.data.size())
        self.q1_fc3.weight.data = fanin_init(self.q1_fc3.weight.data.size())
        self.q1_fc4.weight.data = fanin_init(self.q1_fc4.weight.data.size())
        self.q1_fc5.weight.data = fanin_init(self.q1_fc5.weight.data.size())
        self.q1_fc6.weight.data = fanin_init(self.q1_fc6.weight.data.size())
        self.q1_fc7.weight.data.uniform_(-init_w, init_w)
        # Q2
        nn.init.orthogonal_(self.q2_fc1.weight.data)
        self.q2_fc2.weight.data = fanin_init(self.q2_fc2.weight.data.size())
        self.q2_fc3.weight.data = fanin_init(self.q2_fc3.weight.data.size())
        self.q2_fc4.weight.data = fanin_init(self.q2_fc4.weight.data.size())
        self.q2_fc5.weight.data = fanin_init(self.q2_fc5.weight.data.size())
        self.q2_fc6.weight.data = fanin_init(self.q2_fc6.weight.data.size())
        self.q2_fc7.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        state, action = x
        # Q1
        q1 = self.q1_fc1(torch.cat([state, action], -1))
        q1 = self.relu(q1)
        q1 = self.q1_fc2(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc3(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc4(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc5(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc6(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc7(q1)

        # Q2
        q2 = self.q2_fc1(torch.cat([state, action], -1))
        q2 = self.relu(q2)
        q2 = self.q2_fc2(q2)
        q2 = self.relu(q2)
        q2 = self.q2_fc3(q2)
        q2 = self.relu(q2)
        q2 = self.q2_fc4(q2)
        q2 = self.relu(q2)
        q2 = self.q2_fc5(q2)
        q2 = self.relu(q2)
        q2 = self.q2_fc6(q2)
        q2 = self.relu(q2)
        q2 = self.q2_fc7(q2)
        return q1, q2
    
    def Q1(self, x):
        state, action = x
        q1 = self.q1_fc1(torch.cat([state, action], -1))
        q1 = self.relu(q1)
        q1 = self.q1_fc2(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc3(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc4(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc5(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc6(q1)
        q1 = self.relu(q1)
        q1 = self.q1_fc7(q1)
        return q1