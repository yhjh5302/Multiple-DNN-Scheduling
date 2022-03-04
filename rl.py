import gym
import torch
import numpy as np
from A3C.a3c import A3CAgent
from A2C.a2c import A2CAgent
from DQN.dqn import DQN, dqn_train, dqn_test
from DQN.evaluator import Evaluator
from HEFT.heft import HEFT
from util import *
import time
import argparse


def dqn_main(env, args):
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    if args.mode == "train":
        # save RL options
        with open(args.output + "/rl_options.txt", 'w') as file:
            for key, value in args.__dict__.items():
                file.write(key + ": " + str(value) + "\n")

    if args.seed > 0:
        np.random.seed(args.seed)
        env.seed(args.seed)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    print("num_states:", env.observation_space.shape)
    print("num_actions:", env.action_space.shape)

    agent = DQN(env, num_states, num_actions, args)
    evaluate = Evaluator(args.validate_episodes, args.max_timeslot, args.max_step, 1, save_path=args.output)

    if args.mode == 'train':
        args.debug = True
        dqn_train(num_episodes=args.train_episodes, num_timeslots=args.max_timeslot, max_step=args.max_step, validate_episodes=args.validate_episodes, agent=agent, env=env, evaluate=evaluate, output=args.output, warmup=args.warmup, debug=args.debug)
    #elif args.mode == 'test':
    #    dqn_test(args.validate_episodes, args.max_timeslot, args.max_step, agent, env, evaluate, args.resume, visualize=True, debug=args.debug)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))


def a2c_main(env, gamma, lr, GLOBAL_MAX_EPISODE):
    agent = A2CAgent(env, gamma, lr, GLOBAL_MAX_EPISODE)
    agent.train()
    agent.save_model()


def a3c_main(env, layers, gamma, lr, GLOBAL_MAX_EPISODE):
    agent = A3CAgent(env, layers, gamma, lr, GLOBAL_MAX_EPISODE)
    agent.train()
    agent.save_model()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0))

    from dag_env import DAGEnv
    gym.envs.register(id='DAGEnv-v0', entry_point='dag_env:DAGEnv', max_episode_steps=10000, reward_threshold=np.inf)
    env = gym.make("DAGEnv-v0", max_timeslot=3)
    print("observation:", env.observation_space.shape[0])
    print("action:", env.action_space.shape[0])

    # HEFT
    greedy = HEFT(dataset=env.data_set)
    start = time.time()
    x, y = greedy.run_algo()
    env.data_set.system_manager.set_env(deployed_server=x, execution_order=y)
    print("---------- Greedy Algorithm ----------")
    print("x: ", env.data_set.system_manager.deployed_server)
    print("y: ", y)
    print([s.constraint_chk() for s in env.data_set.system_manager.server.values()])
    #print("t: ", env.data_set.system_manager.total_time())
    print("reward: {:.3f}".format(env.data_set.system_manager.get_reward(cur_p_id=-1, timeslot=0)))
    #print("average_reward: {:.3f}".format(sum([env.data_set.system_manager.get_reward(cur_p_id=env.scheduling_lst[i], timeslot=0) for i in range(env.data_set.num_partitions)]) / env.data_set.num_partitions))
    print([env.data_set.system_manager.get_reward(cur_p_id=env.scheduling_lst[i], timeslot=0) for i in range(env.data_set.num_partitions)])
    #print("took: {:.3f} sec".format(time.time() - start))
    print("---------- Greedy Algorithm ----------\n")

    layers = [1024, 1024, 1024, 1024, 512, 512, 512, 512, 512, 512]
    gamma = 0.99
    lr = 1e-6
    GLOBAL_MAX_EPISODE = 50000
    a3c_main(env, layers, gamma, lr, GLOBAL_MAX_EPISODE)

    # parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    # parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    # parser.add_argument('--env', default='DAGEnv-v0', type=str, help='open-ai gym environment')
    # parser.add_argument('--hidden1', default=512, type=int, help='hidden num of 1st fully connect layer')
    # parser.add_argument('--hidden2', default=512, type=int, help='hidden num of 2nd fully connect layer')
    # parser.add_argument('--hidden3', default=512, type=int, help='hidden num of 3rd fully connect layer')
    # parser.add_argument('--lr', default=1e-6, type=float, help='learning rate')
    # parser.add_argument('--discount', default=0.99, type=float, help='')
    # parser.add_argument('--warmup', default=64, type=int, help='how many episodes without training but only filling the replay memory')
    # parser.add_argument('--bsize', default=64, type=int, help='minibatch size')
    # parser.add_argument('--rmsize', default=1000, type=int, help='memory size')
    # parser.add_argument('--window_length', default=1, type=int, help='')
    # parser.add_argument('--tau', default=0.005, type=float, help='moving average for target network')
    # parser.add_argument('--train_episodes', default=10000, type=int, help='how many episodes to train')
    # parser.add_argument('--max_timeslot', default=1, type=int, help='how many timeslots do we consider')
    # parser.add_argument('--max_step', default=env.data_set.num_partitions, type=int, help='how many steps to perform during each timeslot')
    # parser.add_argument('--validate_episodes', default=1, type=int, help='how many episodes to perform during validate experiment')
    # parser.add_argument('--output', default='output', type=str, help='')
    # parser.add_argument('--debug', dest='debug', action='store_true')
    # parser.add_argument('--init_w', default=0.003, type=float, help='')
    # parser.add_argument('--seed', default=-1, type=int, help='')
    # parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    # args = parser.parse_args()
    # dqn_main(env, args)