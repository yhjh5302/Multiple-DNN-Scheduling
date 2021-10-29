import gym
import torch
import numpy as np
from A3C.a3c import A3CAgent

def train(env, gamma, p_lr, v_lr, GLOBAL_MAX_EPISODE):
    agent = A3CAgent(env, gamma, p_lr, v_lr, GLOBAL_MAX_EPISODE)
    agent.train()
    agent.save_model()


if __name__ == "__main__":
    gamma = 0.99
    v_lr = 5e-5
    p_lr = 5e-5
    GLOBAL_MAX_STEP = 30
    GLOBAL_MAX_TIMESLOT = 1
    GLOBAL_MAX_EPISODE = 100000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0))

    from dag_env import DAGEnv
    gym.envs.register(id='DAGEnv-v0', entry_point='dag_env:DAGEnv', max_episode_steps=GLOBAL_MAX_EPISODE, reward_threshold=np.inf)
    env = gym.make("DAGEnv-v0", max_timeslot=GLOBAL_MAX_TIMESLOT, max_step=GLOBAL_MAX_STEP)
    print("observation:", env.observation_space.shape[0])
    print("action:", env.action_space.shape[0])

    train(env, gamma, p_lr, v_lr, GLOBAL_MAX_EPISODE)