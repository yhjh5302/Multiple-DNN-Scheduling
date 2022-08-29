import torch
import torch.optim as optim
import torch.multiprocessing as mp  
import gym

from A3C.models import ValueNetwork, PolicyNetwork
from A3C.worker import Worker


class A3CAgent:
    def __init__(self, env, layers, gamma, lr, global_max_episode):
        mp.set_start_method('spawn')
        self.env = env

        self.gamma = gamma
        self.lr = lr
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.num_servers = self.env.data_set.num_servers

        self.global_value_network = ValueNetwork(layers=layers, input_dim=self.obs_dim, output_dim=1)
        self.global_value_network.share_memory()
        self.global_value_optimizer = optim.Adam(self.global_value_network.parameters(), lr=lr)

        self.global_policy_network = PolicyNetwork(layers=layers, input_dim=self.obs_dim, output_dim=self.num_servers)
        self.global_policy_network.share_memory()
        self.global_policy_optimizer = optim.Adam(self.global_policy_network.parameters(), lr=lr)
        
        self.workers = [Worker(i, env, layers, self.gamma, self.global_value_network, self.global_policy_network, self.global_value_optimizer, self.global_policy_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE) for i in range(4)]
    
    def train(self):
        print("Training on {} cores and {} workers".format(mp.cpu_count(), len(self.workers)))
        input("Enter to start")

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]
    
    def save_model(self):
        torch.save(self.global_value_network.state_dict(), "a3c_value_model.pth")
        torch.save(self.global_policy_network.state_dict(), "a3c_policy_model.pth")