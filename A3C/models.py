import torch
import torch.nn as nn
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=(5,10), stride=1, padding=2),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,10), stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,8), stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,6), stride=1, padding=1),
                        nn.ReLU(),
                    )

        self.actor_fc1 = nn.Linear(7*10*128, 4096)
        self.actor_fc2 = nn.Linear(4096, 4096)
        self.actor_fc3 = nn.Linear(4096, 4096)
        self.actor_fc4 = nn.Linear(4096, output_dim)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, state):
        x = self.conv(state)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        logits = self.actor_fc1(x)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.actor_fc2(logits)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.actor_fc3(logits)
        logits = self.relu(logits)
        logits = self.dropout(logits)
        logits = self.actor_fc4(logits)

        return logits


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=(5,10), stride=1, padding=2),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,10), stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,8), stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,6), stride=1, padding=1),
                        nn.ReLU(),
                    )

        self.critic_fc1 = nn.Linear(7*10*128, 4096)
        self.critic_fc2 = nn.Linear(4096, 4096)
        self.critic_fc3 = nn.Linear(4096, 4096)
        self.critic_fc4 = nn.Linear(4096, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, state):
        x = self.conv(state)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        values = self.critic_fc1(x)
        values = self.relu(values)
        values = self.dropout(values)
        values = self.critic_fc2(values)
        values = self.relu(values)
        values = self.dropout(values)
        values = self.critic_fc3(values)
        values = self.relu(values)
        values = self.dropout(values)
        values = self.critic_fc4(values)

        return values