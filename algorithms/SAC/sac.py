import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from SAC.model import (QNetworkLSTM, SAC_PolicyNetworkLSTM, ReplayBufferLSTM)
from SAC.util import *

from IPython.display import clear_output
import matplotlib.pyplot as plt


def SAC(num_states, num_actions, env, dataset):
    max_episodes = 10000
    num_actions = num_actions
    num_states  = num_states
    
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
    parser.add_argument('--train', dest='train', action='store_true', default=True)
    parser.add_argument('--test', dest='test', action='store_true', default=False)
    args = parser.parse_args()

    # hyper-parameters for RL training
    batch_size = 64
    update_itr = 1
    hidden_dim = 512
    AUTO_ENTROPY=True
    DETERMINISTIC=False
    rewards = []
    model_path = './model/sac_v2_lstm'

    replay_buffer_size = 1000
    replay_buffer = ReplayBufferLSTM(replay_buffer_size)
    sac_trainer = SAC_Trainer(replay_buffer, num_states, num_actions, hidden_dim=hidden_dim, action_range=1.)

    if args.train:
        # training loop
        for eps in range(max_episodes):
            start = time.time()
            state, last_action = env.reset()

            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)             
            
            for step in range(env.max_step):
                hidden_in = hidden_out
                action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in, deterministic = DETERMINISTIC)
                next_state, reward, done, _ = env.next_step(action)

                if step == 0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out
                episode_state.append(state)
                episode_action.append(action)
                episode_last_action.append(last_action)
                episode_reward.append(reward)
                episode_next_state.append(next_state)
                episode_done.append(done) 

                state = next_state
                last_action = action
                
                # update
                if len(replay_buffer) > batch_size:
                    for i in range(update_itr):
                        _ = sac_trainer.update(batch_size, reward_scale=10., auto_entropy=AUTO_ENTROPY, target_entropy=-1.*num_actions)

                if done:
                    break

            replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action, episode_reward, episode_next_state, episode_done)

            if eps % 20 == 0 and eps > 0: # plot and model saving interval
                plot(rewards)
                np.save('rewards_lstm', rewards)
                sac_trainer.save_model(model_path)
            print("episode took:", time.time() - start)
            print('Episode: ', eps, '| Episode Reward: ', np.sum(episode_reward), max(dataset.system_manager.total_time_dp()))
            rewards.append(np.sum(episode_reward))
        sac_trainer.save_model(model_path)

    if args.test:
        sac_trainer.load_model(model_path)
        for eps in range(10):
            state =  env.reset()
            episode_reward = 0
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda(), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).cuda())  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            
            for step in range(env.max_step):
                hidden_in = hidden_out
                action, hidden_out = sac_trainer.policy_net.get_action(state, last_action, hidden_in, deterministic = DETERMINISTIC)
                next_state, reward, done, _ = env.step(action)

                last_action = action
                episode_reward += reward
                state=next_state

            print('Episode: ', eps, '| Episode Reward: ', episode_reward)


def plot(rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('sac_v2_lstm.png')


class SAC_Trainer():
    def __init__(self, replay_buffer, state_dim, action_dim, hidden_dim, action_range):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.replay_buffer = replay_buffer
        print("state_dim", state_dim, "action_dim", action_dim, "device", self.device)

        self.soft_q_net1 = QNetworkLSTM(state_dim, action_dim, hidden_dim).to(self.device)
        self.soft_q_net2 = QNetworkLSTM(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_soft_q_net1 = QNetworkLSTM(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_soft_q_net2 = QNetworkLSTM(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_net = SAC_PolicyNetworkLSTM(state_dim, action_dim, hidden_dim, action_range).to(self.device)
        self.log_alpha = torch.zeros(1, dtype=torch.float32, requires_grad=True, device=self.device)
        print('Soft Q Network (1,2): ', self.soft_q_net1)
        print('Policy Network: ', self.policy_net)

        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        self.soft_q_criterion1 = nn.MSELoss()
        self.soft_q_criterion2 = nn.MSELoss()

        soft_q_lr = 3e-4
        policy_lr = 1e-4
        alpha_lr  = 3e-4

        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=soft_q_lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    
    def update(self, batch_size, reward_scale=10., auto_entropy=True, target_entropy=-2, gamma=0.99, soft_tau=5e-3):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)

        state           = torch.FloatTensor(state).to(self.device)
        next_state      = torch.FloatTensor(next_state).to(self.device)
        action          = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward          = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  # reward is single value, unsqueeze() to add one dim to be [reward] at the sample dim;
        done            = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)

        predicted_q_value1, _ = self.soft_q_net1(state, action, last_action, hidden_in)
        predicted_q_value2, _ = self.soft_q_net2(state, action, last_action, hidden_in)
        new_action, log_prob, z, mean, log_std, _ = self.policy_net.evaluate(state, last_action, hidden_in)
        new_next_action, next_log_prob, _, _, _, _ = self.policy_net.evaluate(next_state, action, hidden_out)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem
        # Updating alpha wrt entropy
        # alpha = 0.0  # trade-off between exploration (max entropy) and exploitation (max Q) 
        if auto_entropy is True:
            alpha_loss = -(self.log_alpha * (log_prob + target_entropy).detach()).mean()
            # print('alpha loss: ',alpha_loss)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = 1.
            alpha_loss = 0

    # Training Q Function
        predict_target_q1, _ = self.target_soft_q_net1(next_state, new_next_action, action, hidden_out)
        predict_target_q2, _ = self.target_soft_q_net2(next_state, new_next_action, action, hidden_out)
        target_q_min = torch.min(predict_target_q1, predict_target_q2) - self.alpha * next_log_prob
        target_q_value = reward + (1 - done) * gamma * target_q_min # if done==1, only reward
        q_value_loss1 = self.soft_q_criterion1(predicted_q_value1, target_q_value.detach())  # detach: no gradients for the variable
        q_value_loss2 = self.soft_q_criterion2(predicted_q_value2, target_q_value.detach())


        self.soft_q_optimizer1.zero_grad()
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()
        self.soft_q_optimizer2.zero_grad()
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()  

    # Training Policy Function
        predict_q1, _= self.soft_q_net1(state, new_action, last_action, hidden_in)
        predict_q2, _ = self.soft_q_net2(state, new_action, last_action, hidden_in)
        predicted_new_q_value = torch.min(predict_q1, predict_q2)
        policy_loss = (self.alpha * log_prob - predicted_new_q_value).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    # Soft update the target value net
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )
        return predicted_new_q_value.mean()

    def save_model(self, path):
        torch.save(self.soft_q_net1.state_dict(), path+'_q1')
        torch.save(self.soft_q_net2.state_dict(), path+'_q2')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.soft_q_net1.load_state_dict(torch.load(path+'_q1'))
        self.soft_q_net2.load_state_dict(torch.load(path+'_q2'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))

        self.soft_q_net1.eval()
        self.soft_q_net2.eval()
        self.policy_net.eval()