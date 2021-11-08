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

        self.actor_fc = nn.Sequential(
                        nn.Linear(7*10*128, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(4096, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, output_dim),
                    )
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state):
        x = self.conv(state)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        logits = self.actor_fc(x)
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

        self.critic_fc = nn.Sequential(
                        nn.Linear(7*10*128, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(4096, 4096),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(4096, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 2048),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(2048, 1),
                    )
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state):
        x = self.conv(state)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        values = self.critic_fc(x)
        return values