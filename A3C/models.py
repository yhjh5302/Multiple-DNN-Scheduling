import torch
import torch.nn as nn
import numpy as np


class ContainerPolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_containers, num_servers):
        super(ContainerPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=input_dim, out_channels=96, kernel_size=(3,6), stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,6), stride=1, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,6), stride=1, padding=0),
                        nn.ReLU(),
                    )

        self.actor_container_fc1 = nn.Linear(17*6*96, 4096)
        self.actor_container_fc2 = nn.Linear(4096, 4096)
        self.actor_container_fc3 = nn.Linear(4096, 4096)
        self.actor_container_fc4 = nn.Linear(4096, num_containers)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state):
        x = self.conv(state)
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

        return container_logits


class ServerPolicyNetwork(nn.Module):
    def __init__(self, input_dim, num_containers, num_servers):
        super(ServerPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=input_dim, out_channels=96, kernel_size=(3,6), stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,6), stride=1, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,6), stride=1, padding=0),
                        nn.ReLU(),
                    )

        self.actor_server_fc1 = nn.Linear(17*6*96+num_containers, 4096)
        self.actor_server_fc2 = nn.Linear(4096, 4096)
        self.actor_server_fc3 = nn.Linear(4096, 4096)
        self.actor_server_fc4 = nn.Linear(4096, num_servers)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state, container_action):
        x = self.conv(state)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        server_logits = self.actor_server_fc1(torch.cat([x, container_action], 1))
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc2(server_logits)
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc3(server_logits)
        server_logits = self.relu(server_logits)
        server_logits = self.dropout(server_logits)
        server_logits = self.actor_server_fc4(server_logits)

        return server_logits


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, num_containers, num_servers):
        super(ValueNetwork, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels=input_dim, out_channels=96, kernel_size=(3,6), stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,6), stride=1, padding=0),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3,6), stride=1, padding=0),
                        nn.ReLU(),
                    )

        self.critic_fc1 = nn.Linear(17*6*96+num_containers+num_servers, 4096)
        self.critic_fc2 = nn.Linear(4096, 4096)
        self.critic_fc3 = nn.Linear(4096, 4096)
        self.critic_fc4 = nn.Linear(4096, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state, action):
        x = self.conv(state)
        x = self.dropout(x)

        x = torch.flatten(x, 1)

        values = self.critic_fc1(torch.cat([x, action], 1))
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