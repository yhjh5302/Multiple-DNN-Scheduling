#!/usr/bin/env python3 

import numpy as np
import argparse
from copy import deepcopy
import torch
import gym

from evaluator import Evaluator
from ddpg import DDPG
from util import *

import time

from ounoise import OUNoise
from parameter_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric


def train(num_episodes, num_timeslots, max_step, validate_episodes, agent, ounoise, param_noise, env, evaluate, output, debug=False):
    agent.is_training = True
    for episode in range(num_episodes):
        episode_reward = 0.
        episode_loss = 0.

        for timeslot in range(num_timeslots):
            observation = None
            timeslot_reward = 0.
            timeslot_loss = 0.

            ounoise.scale_update(episode) # use in select_action
            #agent.perturb_actor_parameters(param_noise)
            #perturbed_actions = list()
            #unperturbed_actions = list()

            for step in range(max_step):
                # reset if it is the start of episode
                if observation is None:
                    observation = deepcopy(env.reset())
                    agent.reset(observation)

                # agent pick action ...
                if episode * num_timeslots * max_step + timeslot * max_step + step < args.warmup:
                    action = agent.random_action(state=observation)
                else:
                    action = agent.select_action(state=observation, ounoise=ounoise, param_noise=None)
                    #perturbed_actions.append(to_numpy(torch.cat(agent.actor_perturbed(to_tensor(observation)), -1)))
                    #unperturbed_actions.append(to_numpy(torch.cat(agent.actor(to_tensor(observation)), -1)))

                # env response with next_observation, reward, terminate_info
                observation2, reward, done, info = env.step(action)
                observation2 = deepcopy(observation2)
                if step >= max_step - 1:
                    done = True

                # agent observe and update policy
                agent.observe(reward, observation2, done)
                if episode * num_timeslots * max_step + timeslot * max_step + step >= args.warmup:
                    loss = agent.update_policy()
                    timeslot_loss += loss

                # update
                observation = deepcopy(observation2)
                timeslot_reward += reward

                if done:
                    break

            # after timeslot - update reward, loss
            episode_reward += timeslot_reward
            episode_loss += timeslot_loss

            # after timeslot - update battery
            env.after_timeslot()

            # after timeslot - update noise
            #if len(perturbed_actions) > 0 and len(unperturbed_actions) > 0:
            #    ddpg_dist = ddpg_distance_metric(np.array(perturbed_actions), np.array(unperturbed_actions))
            #    param_noise.adapt(ddpg_dist)

            if debug:
                prGreen('#{}: timeslot:{} avg_timeslot_reward:{} loss:{}'.format(episode, timeslot, timeslot_reward / max_step, timeslot_loss / max_step))
                #env.PrintState(observation) # for debug
                print("Y", env.data_set.system_manager._y) # for debug
                #print("perturbed_actions", np.mean(np.array(perturbed_actions), axis=0))
                #print("unperturbed_actions", np.mean(np.array(unperturbed_actions), axis=0))
                #print("stddev", param_noise.current_stddev)

        # after episode - reset environment
        env.after_episode()

        if debug:
            prCyan('#{}: avg_episode_reward:{} loss:{}'.format(episode, episode_reward / max_step / num_timeslots, episode_loss / max_step / num_timeslots))

        # [optional] save intermideate model
        if episode % int(num_episodes / 5) == 0:
            agent.save_model(output)

        # [optional] evaluate
        if evaluate is not None and validate_episodes > 0 and episode % validate_episodes == 0:
            policy = lambda x: agent.select_action(x)
            validate_reward = evaluate(env, policy, debug=debug, visualize=False)
            if debug:
                prYellow('[Evaluate] Episode_{:04d}: mean_reward:{}'.format(episode, validate_reward))


def test(num_episodes, num_timeslots, max_step, agent, env, evaluate, model_path, visualize=True, debug=False):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x)

    for i in range(num_episodes):
        validate_reward = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        if debug:
            prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward / 1000))


if __name__ == "__main__":
    start_time = time.time()

    # load environment
    from dag_env import DAGEnv
    gym.envs.register(id='DAGEnv-v0', entry_point='dag_env:DAGEnv', max_episode_steps=10000, reward_threshold=np.inf)

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='DAGEnv-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--hidden1', default=2048, type=int, help='hidden num of 1st fully connect layer')
    parser.add_argument('--hidden2', default=2048, type=int, help='hidden num of 2nd fully connect layer')
    parser.add_argument('--hidden3', default=2048, type=int, help='hidden num of 3rd fully connect layer')
    parser.add_argument('--hidden4', default=2048, type=int, help='hidden num of 4th fully connect layer')
    parser.add_argument('--hidden5', default=1024, type=int, help='hidden num of 5th fully connect layer')
    parser.add_argument('--hidden6', default=1024, type=int, help='hidden num of 6th fully connect layer')
    parser.add_argument('--learning_rate', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--policy_learning_rate', default=0.00005, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--policy_freq', default=2, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--warmup', default=64, type=int, help='how many episodes without training but only filling the replay memory')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    parser.add_argument('--rmsize', default=65, type=int, help='memory size')
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.005, type=float, help='moving average for target network')
    parser.add_argument('--train_episodes', default=1000, type=int, help='how many episodes to train')
    parser.add_argument('--max_timeslot', default=24, type=int, help='how many timeslots do we consider')
    parser.add_argument('--max_step', default=1, type=int, help='how many steps to perform during each timeslot')
    parser.add_argument('--validate_episodes', default=1, type=int, help='how many episodes to perform during validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='')
    parser.add_argument('--noise_scale', default=0.3, type=float, metavar='G', help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.003, metavar='G', help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N', help='number of episodes with noise (default: 100)')
    parser.add_argument('--initial_stddev', type=int, default=0.01, metavar='N', help='(default: 0.005)')
    parser.add_argument('--adaptation_coefficient', type=int, default=1.05, metavar='N', help='(default: 1.05)')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # parser.add_argument('--l2norm', default=0.01, type=float, help='l2 weight decay') # TODO
    # parser.add_argument('--cuda', dest='cuda', action='store_true') # TODO

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    if args.mode == "train":
        # save RL options
        with open(args.output + "/rl_options.txt", 'w') as file:
            for key, value in args.__dict__.items():
                file.write(key + ": " + str(value) + "\n")

    if torch.cuda.is_available():
        print("Device:", torch.device("cuda:0"), torch.cuda.get_device_name(0))
    else:
        print("Device:", torch.device("cpu"))

    print(args.env)
    env = gym.make(args.env)

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    print("num_states:", env.observation_space.shape)
    print("num_actions:", env.action_space.shape)

    agent = DDPG(env, num_states, num_actions, args)
    evaluate = Evaluator(args.validate_episodes, args.max_timeslot, args.max_step, 1, save_path=args.output)

    # noise initialize
    ounoise = OUNoise(action_dimension=agent.num_actions, noise_scale=args.noise_scale, final_noise_scale=args.final_noise_scale, exploration_end=args.exploration_end)
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=args.initial_stddev, desired_action_stddev=args.noise_scale, adaptation_coefficient=args.adaptation_coefficient)

    if args.mode == 'train':
        train(args.train_episodes, args.max_timeslot, args.max_step, args.validate_episodes, agent, ounoise, param_noise, env, evaluate, args.output, debug=args.debug)
    elif args.mode == 'test':
        test(args.validate_episodes, args.max_timeslot, args.max_step, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))

    end_time = time.time()
    print("Time took:", end_time - start_time, "sec")