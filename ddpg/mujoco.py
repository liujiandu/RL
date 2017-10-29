import gym
import time
import numpy as np

env = gym.make("Humanoid-v1")
env.reset()
for _ in range(200):
    env.render()
    env.step((np.random.random(17)-0.5))
    time.sleep(0.1)
