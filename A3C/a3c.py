import torch
import torch.optim as optim
import torch.multiprocessing as mp  
import gym

from A3C.models import PolicyNetwork, ValueNetwork
from A3C.worker import Worker


class A3CAgent:
    def __init__(self, env, gamma, plr, vlr, global_max_episode):
        mp.set_start_method('spawn')
        self.env = env

        self.gamma = gamma
        self.plr = plr
        self.vlr = vlr
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.num_servers = self.env.data_set.num_servers
        self.num_containers = self.env.data_set.num_containers

        self.obs_dim = self.env.data_set.system_manager.NUM_CHANNEL
        self.action_dim = self.num_containers * 3 # move left, stay, move right

        self.global_policy_network = PolicyNetwork(self.obs_dim, self.action_dim)
        self.global_policy_network.share_memory()
        self.global_policy_optimizer = optim.Adam(self.global_policy_network.parameters(), lr=plr)

        self.global_value_network = ValueNetwork(self.obs_dim)
        self.global_value_network.share_memory()
        self.global_value_optimizer = optim.Adam(self.global_value_network.parameters(), lr=vlr)
        
        #self.workers = [Worker(i, env, self.gamma, self.global_network, self.global_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE) for i in range(int(mp.cpu_count() / 4))]
        self.worker = Worker(0, env, self.gamma, self.global_policy_network, self.global_policy_optimizer, self.global_value_network, self.global_value_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE)
    
    def train(self):
        #print("Training on {} cores and {} workers".format(mp.cpu_count(), len(self.workers)))
        #input("Enter to start")

        #[worker.start() for worker in self.workers]
        #[worker.join() for worker in self.workers]
        self.worker.run()
    
    def save_model(self):
        torch.save(self.global_policy_network.state_dict(), "a3c_policy_model.pth")
        torch.save(self.global_value_network.state_dict(), "a3c_value_model.pth")