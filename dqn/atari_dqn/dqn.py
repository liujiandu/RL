#-----------------------
# File: Deep Q network algorithm 
# Author: Jiandu Liu
# Date: 2017.10.28
#------------------------
import tensorflow as tf
import numpy as np
import random
import cv2
import gym
from collections import deque
import qnet

class DQN(object):
    def __init__(self,nact,
                 observe_time_step = 100,
                 target_update_time_step = 100,
                 frame_per_action=1,
                 epsilon = 0.1,
                 gamma = 0.99,
                 tau = 0.1,
                 batch_size = 32,
                 replay_memory_size = 10000):
        
        #hypercompareter
        self.time_step = 0
        self.nact = nact
        self.observe_time_step = observe_time_step
        self.target_update_time_step = target_update_time_step
        self.frame_per_action = frame_per_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size

        #replay memory
        self.replay_memory = deque()
        
        #input of policy network
        self.action = tf.placeholder(tf.int64, [None])
        self.state = tf.placeholder(tf.float32, [None,80,80,4])
        self.yval = tf.placeholder(tf.float32, [None])
        
        #q_value and loss of policy network
        self.qvals = qnet.qnet(self.state, self.nact, "qnet", reuse=False)
        #tensorflow 0.8
        #self.qval = tf.reduce_sum(tf.mul(self.qvals, tf.cast(tf.one_hot(self.action, self.nact, 1,0),dtype=tf.float32)), reduction_indices=1)
        self.qval = tf.reduce_sum(self.qvals*tf.cast(tf.one_hot(self.action, self.nact, 1,0),dtype=tf.float32), reduction_indices=1)
        self.target_qvals = qnet.qnet(self.state, self.nact, "qtarget_net", reuse=False)
        self.loss = tf.reduce_mean(tf.square(self.qval-self.yval))
        
        #train step
        self.train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
    
        #trainable variables of policy network and target policy network
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qnet")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qtarget_net")
        
        self.target_init_update, self.target_soft_update = self.get_target_update()
        #session
        self.sess = tf.Session()
        
        #initialize network variables
        #self.sess.run(tf.initialize_all_variables()) ##tensorflow 0.8
        self.sess.run(tf.global_variables_initializer()) ##tensorflow 1.1
        self.sess.run(self.target_init_update)


    def init_state(self, obs):
        return np.stack((obs, obs, obs, obs), axis=2)[:,:,:,0]
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.nact)
        else:
            #note: get action by policy network not target policy network
            action = np.argmax(self.sess.run(self.qvals, {self.state:[state]}))
        return action
    
    
    def get_target_update(self):
        target_init_update = []
        target_soft_update = []
        for var, target_var in zip(self.vars, self.target_vars):
            target_init_update.append(tf.assign(target_var, var))
            target_soft_update.append(tf.assign(target_var, (1.0-self.tau)*target_var+ self.tau*var))
        return tf.group(*target_init_update), tf.group(*target_soft_update)
    
    def update_policy(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        state_batch = [data[0] for data in batch]
        action_batch = [data[1] for data in batch]
        reward_batch = [data[2] for data in batch]
        next_state_batch = [data[3] for data in batch]
        
        #get yval
        yval_batch = []
        target_qvals = self.sess.run(self.target_qvals, {self.state:next_state_batch})
        for i in range(self.batch_size):
            terminal = batch[i][4]
            if terminal == True:
                yval_batch.append(reward_batch[i])
            else:
                yval_batch.append(reward_batch[i]+self.gamma*np.max(target_qvals[i]))
        
        #update q network
        self.sess.run(self.train_step, {self.state:state_batch, self.action:action_batch, self.yval:yval_batch})
        
        #update target network 
        if self.time_step % self.target_update_time_step == 0:
            self.sess.run(self.target_soft_update)

def preprocess(obs):
    obs = cv2.cvtColor(cv2.resize(obs, (80,80)), cv2.COLOR_BGR2GRAY)
    return obs.reshape((80,80,1))



if __name__ == "__main__":
 
    env = gym.make("Breakout-v0")
    nact = env.action_space.n

    dqn = DQN(nact=nact,
              observe_time_step=100,
              target_update_time_step=100,
              epsilon=0.1,
              gamma = 0.99,
              tau = 0.01,
              batch_size = 32,
              replay_memory_size = 100000)
    
    max_episode=1000
    test_episode = 10
    ave_reward = []

    for episode in range(max_episode):
        print ("episode", episode)
        obs = env.reset()
        dqn.current_state = dqn.init_state(preprocess(obs))
        over = False
        while not over:
            if dqn.time_step % dqn.frame_per_action==0:
                action = dqn.get_action(dqn.current_state)
            else:
                action = action
            obs, reward, over, _ = env.step(action)
            #env.render()
            next_state = np.append(dqn.current_state[:,:,1:], preprocess(obs), axis=2)
            dqn.replay_memory.append((dqn.current_state, action, reward, next_state, over))
            if len(dqn.replay_memory)>dqn.replay_memory_size:
                dqn.replay_memory.popleft()
            if dqn.time_step > dqn.observe_time_step:
                dqn.update_policy()

            dqn.current_state = next_state
            dqn.time_step += 1
        
        #test
        if episode % 10 == 0 and episode>0:
            ave_r = 0
            for i in range(test_episode):
                obs = env.reset()
                current_state = dqn.init_state(preprocess(obs))
                over = False
                while not over:
                    action = dqn.get_action(current_state)
                    obs, reward, over, _ = env.step(action)
                    next_state = np.append(current_state[:,:,1:], preprocess(obs), axis=2)
                    ave_r += reward
                    current_state = next_state
            
            ave_r = ave_r/test_episode
            ave_reward.append(ave_r)
            print ("average reward ", ave_r)
    print("done")

    import matplotlib.pyplot as plt
    plt.plot(range(len(avr_reward)), ave_reward)
    plt.show()

    




