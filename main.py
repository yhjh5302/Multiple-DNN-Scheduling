import gym
import numpy as np

if __name__ == "__main__":
    GLOBAL_MAX_EPISODE = 1000
    MAX_TIMESLOT = 24

    from dag_env import DAGEnv
    gym.envs.register(id='DAGEnv-v0', entry_point='dag_env:DAGEnv', max_episode_steps=GLOBAL_MAX_EPISODE, reward_threshold=np.inf)
    env = gym.make("DAGEnv-v0", max_timeslot=MAX_TIMESLOT)
    print("observation:", env.observation_space.shape[0])
    print("action:", env.action_space.shape[0])

    #Search(env, 100)