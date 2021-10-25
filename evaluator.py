
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
        episode_reward = 0
        result = []

        for timeslot in range(self.num_timeslots):
            observation = None
            timeslot_reward = 0.
            # start episode
            for step in range(self.max_step):
                # reset if it is the start of episode
                if observation is None:
                    observation = env.reset()

                # basic operation, action ,reward, blablabla ...
                action = policy(observation)
                observation, reward, done, info = env.step(action)

                # update
                timeslot_reward += reward

                if done:
                    break

            episode_reward += timeslot_reward

            if env.unwrapped.spec.id == 'TYEnv-v0' or env.unwrapped.spec.id == 'DAGEnv-v0':
                env.after_timeslot()

            if debug:
                prYellow('[Evaluate] #Timeslot{}: timeslot_reward:{}'.format(timeslot, timeslot_reward / self.max_step))
                env.PrintState(observation) # for debug

        if env.unwrapped.spec.id == 'TYEnv-v0' or env.unwrapped.spec.id == 'DAGEnv-v0':
            env.after_episode()

        result.append(episode_reward / self.num_timeslots / self.max_step)

        result = np.array(result).reshape(-1,1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results('{}/validate_reward'.format(self.save_path))
        return np.mean(result)

    def save_results(self, fn):
        if self.results.shape[1] % self.interval == 0:
            results = np.mean(self.results.reshape(1,-1,self.interval), axis=2)

            y = np.mean(results, axis=0)
            error=np.std(results, axis=0)
                        
            x = range(0,results.shape[1]*self.interval,self.interval)
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            ax.errorbar(x, y, yerr=error, fmt='-o')
            plt.savefig(fn+'.png')
            savemat(fn+'.mat', {'reward':results})
            plt.close(fig)