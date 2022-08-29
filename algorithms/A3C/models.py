import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):
    def __init__(self, layers, input_dim, output_dim):
        super(ValueNetwork, self).__init__()
        modules = [nn.Linear(input_dim, layers[0])] + [nn.Linear(layers[i-1], layers[i]) for i in range(1, len(layers))] + [nn.Linear(layers[-1], output_dim)]
        self.layers = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = F.relu(x)
            x = self.layers[i](x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, layers, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        modules = [nn.Linear(input_dim, layers[0])] + [nn.Linear(layers[i-1], layers[i]) for i in range(1, len(layers))] + [nn.Linear(layers[-1], output_dim)]
        self.layers = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = F.relu(x)
            x = self.layers[i](x)
        return x