import tensorflow as tf
import gym
import numpy as np

from ou_noise import OUNoise
from critic import Critic
from actor import Actor
from replay_memory import ReplayMemory


class DDPG(object):
    def __init__(self, 
                 env,
                 actor_lr =0.01,
                 critic_lr = 0.01,
                 tau =0.001,
                 gamma = 0.99, 
                 batch_size = 32,
                 replay_memory_size=100000,
                 replay_memory_start_size=1000):
        
        #hyperparameter
        self.actor_lr =actor_lr
        self.critic_lr = critic_lr
        self.tau =tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_memory_size=replay_memory_size
        self.replay_memory_start_size=replay_memory_start_size
    
        self.sess = tf.Session()    
        #environment
        self.env = env  
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        
        #actor
        self.actor = Actor(self.sess, state_dim=self.state_dim,
                action_dim=self.action_dim, tau=self.tau, lr=self.actor_lr)
        
        #critic
        self.critic = Critic(self.sess, state_dim=self.state_dim,
                action_dim=self.action_dim, tau=self.tau, lr=self.critic_lr)
        
        #replay memory
        self.replay_memory = ReplayMemory(self.replay_memory_size)
        
        #exploration noise
        self.exploration_noise = OUNoise(self.action_dim)

    def train(self):
        minibatch = self.replay_memory.get_batch(self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])
        

        # calculate yval_batch
        next_action_batch = self.actor.target_action(next_state_batch)
        next_qval_batch = self.critic.target_qval(next_state_batch, next_action_batch)
        
        yval_batch = []
        for i in range(self.batch_size):
            if done_batch[i]:
                yval_batch.append(reward_batch[i])
            else:
                yval_batch.append(reward_batch[i]+self.gamma*next_qval_batch[i][0])
        yval_batch = np.reshape(yval_batch,[self.batch_size,1])

        #update critic
        self.critic.update(yval_batch, state_batch, action_batch)

        #update actor
        action_batch_for_update_actor = self.actor.action(state_batch)
        action_gradients_batch = self.critic.gradients(state_batch,
                action_batch_for_update_actor)
        self.actor.update(action_gradients_batch[0], state_batch)

        #update target
        self.sess.run(self.actor.soft_target_updates)
        self.sess.run(self.critic.soft_target_updates)
    
    def noise_action(self, state):
        return self.actor.action(state)+self.exploration_noise.noise()

    
    
