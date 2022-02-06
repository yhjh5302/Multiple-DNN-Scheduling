import gym
import torch
import numpy as np
from A3C.a3c import A3CAgent

def train(env, gamma, lr, GLOBAL_MAX_EPISODE):
    agent = A3CAgent(env, gamma, lr, GLOBAL_MAX_EPISODE)
    agent.train()
    agent.save_model()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0))

    from dag_env import DAGEnv
    gym.envs.register(id='DAGEnv-v0', entry_point='dag_env:DAGEnv', max_episode_steps=10000, reward_threshold=np.inf)
    env = gym.make("DAGEnv-v0", max_timeslot=1)
    print("observation:", env.observation_space.shape[0])
    print("action:", env.action_space.shape[0])

    gamma = 0.99
    lr = 1e-3
    GLOBAL_MAX_EPISODE = 10000

    train(env, gamma, lr, GLOBAL_MAX_EPISODE)