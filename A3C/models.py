import torch
import torch.nn as nn
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=(5,10), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,10), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4,8), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,6), stride=1, padding=1)

        self.actor_fc1 = nn.Linear(7*10*64, 4096)
        self.actor_fc2 = nn.Linear(4096, 4096)
        self.actor_fc3 = nn.Linear(4096, output_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        x = self.actor_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.actor_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.actor_fc3(x)

        return x


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, num_action):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=(5,10), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5,10), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,8), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,6), stride=1, padding=1)

        self.critic_fc1 = nn.Linear(7*10*64+num_action, 4096)
        self.critic_fc2 = nn.Linear(4096, 4096)
        self.critic_fc3 = nn.Linear(4096, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, state, action):
        x = self.conv1(state)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        x = self.critic_fc1(torch.cat([x, action], 1))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.critic_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.critic_fc3(x)

        return x