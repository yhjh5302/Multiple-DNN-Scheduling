import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .util import *


class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space, action_range):
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  
        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:  # Discrete space
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]
        self.action_range = action_range

    def forward(self):
        pass
    
    def evaluate(self):
        pass 
    
    def get_action(self):
        pass

    def sample_action(self,):
        a=torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return self.action_range*a.numpy()


class SAC_PolicyNetworkLSTM(PolicyNetworkBase):
    def __init__(self, state_space, action_space, hidden_size, action_range=1., init_w=3e-3, log_std_min=-20, log_std_max=2):
        super().__init__(state_space, action_space, action_range=action_range)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.hidden_size = hidden_size
        
        self.linear1 = nn.Linear(self._state_dim, hidden_size)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, self._action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, self._action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)


    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = F.relu(self.linear1(state))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = F.relu(self.linear2(lstm_branch))
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 
        x = F.relu(self.linear3(merged_branch))
        x = F.relu(self.linear4(x))
        x = x.permute(1,0,2)  # permute back

        mean    = self.mean_linear(x)
        # mean    = F.leaky_relu(self.mean_linear(x))
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std, lstm_hidden
    
    def evaluate(self, state, last_action, hidden_in, epsilon=1e-6):
        '''
        generate sampled action with state as input wrt the policy network;
        '''
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp() # no clip in evaluation, clip affects gradients flow
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape)
        action_0 = torch.tanh(mean + std * z.cuda())  # TanhNormal distribution as actions; reparameterization trick
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(mean + std * z.cuda()) - torch.log(
            1. - action_0.pow(2) + epsilon) - np.log(self.action_range)
        # both dims of normal.log_prob and -log(1-a**2) are (N,dim_of_action);
        # the Normal.log_prob outputs the same dim of input features instead of 1 dim probability,
        # needs sum up across the features dim to get 1 dim prob; or else use Multivariate Normal.
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z, mean, log_std, hidden_out

    def get_action(self, state, last_action, hidden_in, deterministic=True):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda()  # increase 2 dims to match with training data
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        mean, log_std, hidden_out = self.forward(state, last_action, hidden_in)
        std = log_std.exp()
        
        normal = Normal(0, 1)
        z = normal.sample(mean.shape).cuda()
        action = self.action_range * torch.tanh(mean + std * z)

        action = self.action_range * torch.tanh(mean).detach().cpu().numpy() if deterministic else \
        action.detach().cpu().numpy()
        return action[0][0], hidden_out


class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_dim, activation):
        super(ValueNetworkBase, self).__init__()
        self._state_dim = state_dim
        self.activation = activation

    def forward(self):
        pass


class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_dim, action_dim, activation):
        super().__init__(state_dim, activation)
        self._action_dim = action_dim


class QNetwork(QNetworkBase):
    def __init__(self, state_dim, action_dim, hidden_dim, activation=F.relu):
        super().__init__(state_dim, action_dim, activation)

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1) # the dim 0 is number of samples
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        x = self.linear4(x)
        return x


class QNetworkLSTM(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows two-branch structure as in paper: 
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization
    """
    def __init__(self, state_dim, action_dim, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_dim, action_dim, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.linear2 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear4.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # branch 1
        fc_branch = torch.cat([state, action], -1) 
        fc_branch = self.activation(self.linear1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1) 
        lstm_branch = self.activation(self.linear2(lstm_branch))  # linear layer for 3d input only applied on the last dim
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch=torch.cat([fc_branch, lstm_branch], -1) 

        x = self.activation(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)   


class QNetworkLSTM2(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+2*self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # single branch
        x = torch.cat([state, action, last_action], -1) 
        x = self.activation(self.linear1(x))
        x, lstm_hidden = self.lstm1(x, hidden_in)  # no activation after lstm
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell) 


class ReplayBufferLSTM:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst=[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)