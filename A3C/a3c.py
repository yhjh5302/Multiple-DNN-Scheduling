import torch
import torch.optim as optim
import torch.multiprocessing as mp  
import gym

from A3C.models import ContainerPolicyNetwork, ServerPolicyNetwork, ValueNetwork
from A3C.worker import Worker


class A3CAgent:
    def __init__(self, env, gamma, plr, vlr, buffer_size, batch_size, global_max_episode):
        mp.set_start_method('spawn')
        self.env = env

        self.gamma = gamma
        self.plr = plr
        self.vlr = vlr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.obs_dim = self.env.data_set.system_manager.NUM_CHANNEL
        self.action_dim = self.env.data_set.num_servers

        self.num_servers = self.env.data_set.num_servers
        self.num_containers = self.env.data_set.num_containers

        self.global_container_policy_network = ContainerPolicyNetwork(self.obs_dim, self.num_containers, self.num_servers)
        self.global_container_policy_network.share_memory()
        self.global_container_policy_optimizer = optim.Adam(self.global_container_policy_network.parameters(), lr=plr)

        self.global_server_policy_network = ServerPolicyNetwork(self.obs_dim, self.num_containers, self.num_servers)
        self.global_server_policy_network.share_memory()
        self.global_server_policy_optimizer = optim.Adam(self.global_server_policy_network.parameters(), lr=plr)

        self.global_value_network = ValueNetwork(self.obs_dim, self.num_containers, self.num_servers)
        self.global_value_network.share_memory()
        self.global_value_optimizer = optim.Adam(self.global_value_network.parameters(), lr=vlr)
        
        #self.workers = [Worker(i, env, self.gamma, buffer_size, batch_size, self.global_network, self.global_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE) for i in range(int(mp.cpu_count() / 4))]
        self.worker = Worker(0, env, self.gamma, buffer_size, batch_size, self.global_container_policy_network, self.global_container_policy_optimizer, self.global_server_policy_network, self.global_server_policy_optimizer, self.global_value_network, self.global_value_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE)
    
    def train(self):
        #print("Training on {} cores and {} workers".format(mp.cpu_count(), len(self.workers)))
        #input("Enter to start")

        #[worker.start() for worker in self.workers]
        #[worker.join() for worker in self.workers]
        self.worker.run()
    
    def save_model(self):
        torch.save(self.global_container_policy_network.state_dict(), "a3c_container_policy_model.pth")
        torch.save(self.global_server_policy_network.state_dict(), "a3c_server_policy_model.pth")
        torch.save(self.global_value_network.state_dict(), "a3c_value_model.pth")