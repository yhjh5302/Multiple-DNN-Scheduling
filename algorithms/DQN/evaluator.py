import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from util import *
from copy import deepcopy


class Evaluator(object):

    def __init__(self, num_episodes, num_timeslots, max_step, interval, save_path=''):
        self.num_episodes = num_episodes
        self.num_timeslots = num_timeslots
        self.max_step = max_step
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes,0)

    def __call__(self, env, policy, debug=False, visualize=False, save=True):

        self.is_training = False
        result = []

        for timeslot in range(self.num_timeslots):
            observation = None

            # start episode
            for step in range(self.max_step):
                # reset if it is the start of episode
                if observation is None:
                    observation = env.reset()

                # basic operation, action ,reward, blablabla ...
                action = policy(observation, step)
                observation, reward, done, info = env.step(action)

                if done:
                    break

            if debug:
                prYellow('[Evaluate] #Timeslot{}: timeslot_reward:{}'.format(timeslot, reward))
                print(env.data_set.system_manager.deployed_server) # for debug

        result.append(reward)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):

        y = np.mean(self.results, axis=0)
        error=np.std(self.results, axis=0)
                    
        x = range(0,self.results.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':self.results})
        plt.close(fig)