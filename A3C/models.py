import torch
import torch.nn as nn
import numpy as np


class Network(nn.Module):
    def __init__(self, input_dim, num_containers, num_servers):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=64, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=1)

        #self.lstm = nn.LSTMCell(27*7*64, 4096)

        self.actor_container_fc1 = nn.Linear(24*4*128, 4096)
        self.actor_container_fc2 = nn.Linear(4096, 4096)
        self.actor_container_fc3 = nn.Linear(4096, 1024)
        self.actor_container_fc4 = nn.Linear(1024, num_containers)

        self.actor_server_fc1 = nn.Linear(24*4*128+num_containers, 4096)
        self.actor_server_fc2 = nn.Linear(4096, 4096)
        self.actor_server_fc3 = nn.Linear(4096, 1024)
        self.actor_server_fc4 = nn.Linear(1024, num_servers)

        self.critic_fc1 = nn.Linear(24*4*128+num_containers+num_servers, 4096)
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

        container_logits = self.actor_container_fc1(x)
        container_logits = self.relu(container_logits)
        container_logits = self.dropout(container_logits)
        container_logits = self.actor_container_fc2(container_logits)
        container_logits = self.relu(container_logits)
        container_logits = self.dropout(container_logits)
        container_logits = self.actor_container_fc3(container_logits)
        container_logits = self.relu(container_logits)
        container_logits = self.dropout(container_logits)
        container_logits = self.actor_container_fc4(container_logits)

        server_logits = self.actor_server_fc1(torch.cat([x,container_logits], 1))
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc2(server_logits)
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc3(server_logits)
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc4(server_logits)

        values = self.critic_fc1(torch.cat([x,container_logits,server_logits], 1))
        values = self.relu(values)
        values = self.dropout(values)
        values = self.critic_fc2(values)
        values = self.relu(values)
        values = self.dropout(values)
        values = self.critic_fc3(values)
        values = self.relu(values)
        values = self.dropout(values)
        values = self.critic_fc4(values)

        return container_logits, server_logits, values

    def forward_logits(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        container_logits = self.actor_container_fc1(x)
        container_logits = self.relu(container_logits)
        container_logits = self.dropout(container_logits)
        container_logits = self.actor_container_fc2(container_logits)
        container_logits = self.relu(container_logits)
        container_logits = self.dropout(container_logits)
        container_logits = self.actor_container_fc3(container_logits)
        container_logits = self.relu(container_logits)
        container_logits = self.dropout(container_logits)
        container_logits = self.actor_container_fc4(container_logits)

        server_logits = self.actor_server_fc1(torch.cat([x,container_logits], 1))
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc2(server_logits)
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc3(server_logits)
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc4(server_logits)

        return container_logits, server_logits