import torch
import torch.optim as optim
import torch.multiprocessing as mp  
import gym

from A3C.models import Network
from A3C.worker import Worker


class A3CAgent:
    def __init__(self, env, gamma, lr, global_max_episode):
        mp.set_start_method('spawn')
        self.env = env

        self.gamma = gamma
        self.lr = lr
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.num_servers = self.env.data_set.num_servers
        self.num_containers = self.env.data_set.num_containers

        self.global_network = Network(self.obs_dim, self.action_dim)
        self.global_network.share_memory()
        self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr)
        
        self.workers = [Worker(i, env, self.gamma, self.global_network, self.global_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE) for i in range(int(mp.cpu_count() / 12))]
    
    def train(self):
        print("Training on {} cores and {} workers".format(mp.cpu_count(), len(self.workers)))
        input("Enter to start")

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]
    
    def save_model(self):
        torch.save(self.global_network.state_dict(), "a3c_model.pth")