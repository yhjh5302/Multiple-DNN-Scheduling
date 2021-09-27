import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, noise_scale, final_noise_scale, exploration_end, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.noise_scale = noise_scale
        self.final_noise_scale = final_noise_scale
        self.exploration_end = exploration_end

        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def scale_update(self, episode):
        self.scale = (self.noise_scale - self.final_noise_scale) * max(0, self.exploration_end - episode) / self.exploration_end + self.final_noise_scale
        self.reset()

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale