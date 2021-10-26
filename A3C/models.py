import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        #self.lstm = nn.LSTMCell(27*7*64, 4096)

        self.actor_fc1 = nn.Linear(27*7*64, 4096)
        self.actor_fc2 = nn.Linear(4096, 4096)
        self.actor_fc3 = nn.Linear(4096, 1024)
        self.actor_fc4 = nn.Linear(1024, output_dim)

        self.critic_fc1 = nn.Linear(27*7*64, 4096)
        self.critic_fc2 = nn.Linear(4096, 4096)
        self.critic_fc3 = nn.Linear(4096, 1024)
        self.critic_fc4 = nn.Linear(1024, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        logits = self.actor_fc1(x)
        logits = self.relu(logits)
        logits = self.actor_fc2(logits)
        logits = self.relu(logits)
        logits = self.actor_fc3(logits)
        logits = self.relu(logits)
        logits = self.actor_fc4(logits)

        values = self.critic_fc1(x)
        values = self.relu(values)
        values = self.critic_fc2(values)
        values = self.relu(values)
        values = self.critic_fc3(values)
        values = self.relu(values)
        values = self.critic_fc4(values)

        return logits, values

    def forward_logits(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        logits = self.actor_fc1(x)
        logits = self.relu(logits)
        logits = self.actor_fc2(logits)
        logits = self.relu(logits)
        logits = self.actor_fc3(logits)
        logits = self.relu(logits)
        logits = self.actor_fc4(logits)

        return logits