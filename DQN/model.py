import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Network(nn.Module):
    def __init__(self, num_states, num_container, hidden1=256, hidden2=256, hidden3=256, init_w=3e-3):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(num_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, num_container)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        nn.init.orthogonal_(self.fc1.weight.data)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        action = self.fc1(state)
        action = self.relu(action)
        action = self.fc2(action)
        action = self.relu(action)
        action = self.fc3(action)
        action = self.relu(action)
        action = self.fc4(action)
        return action