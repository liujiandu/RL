#-----------------------
# File: Atari game play
# Atuhor: Jiandu Liu
# Date: 2017.10.28
#-----------------------

import gym
import numpy as np
import time
import sys

def play_game(game_name):
    env = gym.make(game_name)
    while(1):
        env.reset()
        over = False
        while not over:
            env.render()
            time.sleep(0.09)
            act = np.random.randint(env.action_space.n)
            obs, reward, over, lives = env.step(act)
            print ("action: ", act) 
        env.render()

if __name__=="__main__":
    play_game(sys.argv[1])
