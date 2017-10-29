from memory import Memory
from models import Actor, Critic
import gym
import tensorflow as tf

def run():
    env = gym.make("Humanoid-v1")
    act_shape = env.observation_space.shape
    obs_shape = env.action_space.shape
    memory = Memory(limit=int(1e2), action_shape = act_shape, observation_shape=obs_shape)
    critic = Critic(layer_norm = True)
    
    actor = Actor(act_shape[0],layer_norm= True)
    obs = env.reset()
    print obs.shape
    state = tf.placeholder(tf.float32, [None,obs_shape[0]])
    
    sess = tf.Session(actor(state),{state:[obs]})
    sess.run()

if __name__ =="__main__":
    run()
