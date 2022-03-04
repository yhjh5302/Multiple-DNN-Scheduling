import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
import pyro.distributions as dist

from DQN.model import Network
from DQN.memory import SequentialMemory
from util import *

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DQN(object):
    def __init__(self, env, num_states, num_actions, args):
        if args.seed > 0:
            self.seed(args.seed)

        self.env = env
        self.num_servers = self.env.data_set.num_servers
        self.num_partitions = self.env.data_set.num_partitions

        self.num_states = num_states
        self.num_actions = num_actions

        net_cfg = {'hidden1':args.hidden1, 'hidden2':args.hidden2, 'hidden3':args.hidden3, 'init_w':args.init_w}

        self.network = Network(self.num_states, self.num_servers, **net_cfg)
        self.network_target = Network(self.num_states, self.num_servers, **net_cfg)
        self.network_optim  = Adam(self.network.parameters(), lr=args.lr)

        hard_update(self.network_target, self.network) # Make sure target is with the same weight

        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount

        # variables
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        self.epsilon = 0.9
        self.max_iteration = 100
        self.iteration = 0
        self.loss_func = nn.MSELoss()

        # cuda
        if USE_CUDA:
            self.cuda()

    def update_policy(self):
        # update the parameters
        if self.iteration % self.max_iteration == 0:
            self.network_target.load_state_dict(self.network.state_dict())
        self.iteration += 1

        # Sample batch
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # Prepare for the target q batch
        q_eval = self.network(to_tensor(state_batch)).gather(1, to_tensor(action_batch, dtype=torch.int64).cuda())
        q_next = self.network_target(to_tensor(next_state_batch)).detach()
        q_target = to_tensor(reward_batch) + self.discount * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.network_optim.zero_grad()
        loss.backward()
        self.network_optim.step()

    def eval(self):
        self.network.eval()
        self.network_target.eval()

    def cuda(self):
        self.network.cuda()
        self.network_target.cuda()

    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done)
            self.s_t = s_t1

    def softmax(self, x):
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x))
        return f_x

    def random_action(self, state, step):
        mask = self.env.get_mask(step=step)
        action = np.random.choice(a=self.num_servers, p=self.softmax(mask))
        self.a_t = action
        return action

    def select_action(self, state, step):
        if np.random.randn() > self.epsilon:
            action = self.random_action(state, step)
        else:
            mask = self.env.get_mask(step=step)
            action = self.network(to_tensor(state)) + to_tensor(mask)
            action = torch.argmax(action).item()
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs

    def load_weights(self, output):
        if output is None:
            return
        self.network.load_state_dict(torch.load('{}/network.pkl'.format(output)))

    def save_model(self,output):
        torch.save(self.network.state_dict(), '{}/network.pkl'.format(output))

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)


def dqn_train(num_episodes, num_timeslots, max_step, validate_episodes, agent, env, evaluate, output, warmup, debug=False):
    from copy import deepcopy
    agent.is_training = True
    for episode in range(num_episodes):

        for timeslot in range(num_timeslots):
            observation = None

            for step in range(max_step):
                # reset if it is the start of episode
                if observation is None:
                    observation = deepcopy(env.reset())
                    agent.reset(observation)

                # agent pick action ...
                if episode * num_timeslots * max_step + timeslot * max_step + step < warmup:
                    action = agent.random_action(state=observation, step=step)
                else:
                    action = agent.select_action(state=observation, step=step)

                # env response with next_observation, reward, terminate_info
                observation2, reward, done, info = env.step(action)
                observation2 = deepcopy(observation2)
                if step >= max_step - 1:
                    done = True

                # agent observe and update policy
                agent.observe(reward, observation2, done)
                if episode * num_timeslots * max_step + timeslot * max_step + step >= warmup:
                    agent.update_policy()

                # update
                observation = deepcopy(observation2)

                if done:
                    break

            # after timeslot - update battery
            #env.after_timeslot()

            if debug:
                prGreen('#{}: timeslot:{} avg_timeslot_reward:{}'.format(episode, timeslot, reward))
                print(env.data_set.system_manager.deployed_server)

        # after episode - reset environment
        #env.after_episode()

        if debug:
            prCyan('#{}: avg_episode_reward:{}'.format(episode, reward))

        # [optional] save intermideate model
        if episode % int(num_episodes / 5) == 0:
            agent.save_model(output)

        # [optional] evaluate
        if evaluate is not None and validate_episodes > 0 and episode % validate_episodes == 0:
            policy = lambda x, y: agent.select_action(x, y)
            validate_reward = evaluate(env, policy, debug=debug, visualize=False)
            if debug:
                prYellow('[Evaluate] Episode_{:04d}: mean_reward:{}'.format(episode, validate_reward))


def dqn_test(num_episodes, num_timeslots, max_step, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x, y: agent.select_action(x, y)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug:
            prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward / 1000))