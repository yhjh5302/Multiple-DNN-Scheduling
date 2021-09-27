import numpy as np
import gym

from svc_algorithm.genetic import *
from dag_env import DAGEnv

import time


def main():
    gym.envs.register(id='DAGEnv-v0', entry_point='dag_env:DAGEnv', max_episode_steps=10000, reward_threshold=np.inf)
    env = gym.make('DAGEnv-v0')

    g = Genetic(env, 100)
    x = g.run_algo(100)
    d._sys_manager.set_x_mat(x)
    # d.system_manager.update_edge_computing_time()
    time = d.system_manager.total_time()
    print("t: ", time)
    reward = g.evaluation(np.array([x]).reshape((1, -1)))
    need_fix_cpu, need_fix_mem, remain_cpu, remain_mem = d.system_manager.constraint_chk(x, d.system_manager._y, inv_opt=True)
    print(need_fix_mem, need_fix_cpu)
    print("x: ", x)
    print("reward: ", reward[0])
    print("normed reward: ", reward[0])


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print("Time took:", end_time - start_time, "sec")