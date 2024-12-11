import ale_py
import numpy as np
from gym.spaces.box import Box
import cv2
import gymnasium as gym

class Normalize(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
    An observation wrapper that normalizes observations using an exponential moving average.

    Attributes:
        x_mean (float): The running mean of the observations.
        x_std (float): The running standard deviation of the observations.
        a (float): Smoothing factor for exponential moving average.
        stepct (int): Count of steps for normalization updates.
    """
        super(Normalize, self).__init__(env)
        self.x_mean = 0
        self.x_std = 0
        self.a = 0.9999
        self.stepct = 0

    def observation(self, obs):
        """Process the observation by normalizing it."""
        self.stepct += 1
        self.x_mean = self.x_mean * self.a + obs.mean() * (1 - self.a)
        self.x_std = self.x_std * self.a + obs.std() * (1 - self.a)

        _mean = self.x_mean / (1 - pow(self.a, self.stepct))
        _std = self.x_std / (1 - pow(self.a, self.stepct))

        norm_output = (obs - _mean) / (_std + 1e-8)
        return norm_output

    def seed(self, seed=None):
        print(f"Setting seed: {seed}")
        self.env.reset(seed=seed)


class DownScale_42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(DownScale_42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, obs):
    
        # resize by half
        frame = obs[34:34 + 160, :160]
        frame = cv2.resize(frame, (80, 80))
        # downscale to 42 x 42
        frame = cv2.resize(frame, (42, 42))
        
        frame = frame.mean(2, keepdims=True)
        frame = frame.astype(np.float32)
        
        # grayscale/normalize
        frame *= (1.0 / 255.0)
        frame = np.moveaxis(frame, -1, 0)
        return frame

    def seed(self, seed=None):
        print(f"Setting seed: {seed}")
        self.env.reset(seed=seed)


