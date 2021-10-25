
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
    def __init__(self, num_states, num_container, num_server, hidden=[1024, 1024, 512], init_w=3e-3):
        super(Actor, self).__init__()

        container_layers = [(num_states, hidden[0])] + [(hidden[i - 1], hidden[i]) for i in range(1, len(hidden))] + [(hidden[-1], num_container)]
        container = [nn.Linear(*l) for l in container_layers]
        self.container = nn.Sequential(*container)

        server_layers = [(num_states, hidden[0])] + [(hidden[i - 1], hidden[i]) for i in range(1, len(hidden))] + [(hidden[-1], num_server)]
        server = [nn.Linear(*l) for l in server_layers]
        self.server = nn.Sequential(*server)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        # container 
        nn.init.orthogonal_(self.container[0].weight.data)
        for i in range(1, len(self.container) - 1):
            self.container[i].weight.data = fanin_init(self.container[i].weight.data.size())
        self.container[-1].weight.data.uniform_(-init_w, init_w)

        # server
        nn.init.orthogonal_(self.server[0].weight.data)
        for i in range(1, len(self.server) - 1):
            self.server[i].weight.data = fanin_init(self.server[i].weight.data.size())
        self.server[-1].weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        # container
        container = self.container[0](x)
        for i in range(1, len(self.container)):
            container = self.relu(container)
            container = self.container[i](container)

        # server
        server = self.server[0](x)
        for i in range(1, len(self.server)):
            server = self.relu(server)
            server = self.server[i](server)

        action = torch.cat([container, server], -1)
        return action

class Critic(nn.Module):
    def __init__(self, num_states, num_actions, hidden=[1024, 1024, 512], init_w=3e-3):
        super(Critic, self).__init__()

        layers = [(num_states + 2, hidden[0])] + [(hidden[i - 1], hidden[i]) for i in range(1, len(hidden))] + [(hidden[-1], 1)]
        q1 = [nn.Linear(*l) for l in layers]
        q2 = [nn.Linear(*l) for l in layers]
        self.q1 = nn.Sequential(*q1)
        self.q2 = nn.Sequential(*q2)

        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        # Q1
        nn.init.orthogonal_(self.q1[0].weight.data)
        for i in range(1, len(self.q1) - 1):
            self.q1[i].weight.data = fanin_init(self.q1[i].weight.data.size())
        self.q1[-1].weight.data.uniform_(-init_w, init_w)

        # Q2
        nn.init.orthogonal_(self.q2[0].weight.data)
        for i in range(1, len(self.q2) - 1):
            self.q2[i].weight.data = fanin_init(self.q2[i].weight.data.size())
        self.q2[-1].weight.data.uniform_(-init_w, init_w)
    
    def forward(self, x):
        state, action = x

        # Q1
        q1 = self.q1[0](torch.cat([state, action], 1))
        for i in range(1, len(self.q1)):
            q1 = self.relu(q1)
            q1 = self.q1[i](q1)

        # Q2
        q2 = self.q2[0](torch.cat([state, action], 1))
        for i in range(1, len(self.q2)):
            q2 = self.relu(q2)
            q2 = self.q2[i](q2)
        return q1, q2
    
    def Q1(self, x):
        state, action = x

        # Q1
        q1 = self.q1[0](torch.cat([state, action], 1))
        for i in range(1, len(self.q1)):
            q1 = self.relu(q1)
            q1 = self.q1[i](q1)
        return q1