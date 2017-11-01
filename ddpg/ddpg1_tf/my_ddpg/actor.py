import tensorflow as tf
import numpy as np
import math

LAYER_SIZE1 = 400
LAYER_SIZE2 = 300

def fc(x, scope, shape, actf=tf.nn.relu):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", shape=shape, 
                initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("b", shape=[shape[1]],
                initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x,w)+b
        return actf(z)

class Actor(object):
    def __init__(self, sess, state_dim, action_dim, lr=0.001, tau=0.01):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.lr = lr

        #create actor network
        self.state_input = tf.placeholder("float", [None,self.state_dim])
        self.action_output = self.create_network(self.state_input, "Actor")
        
        self.target_state_input = tf.placeholder("float", [None,self.state_dim])
        self.target_action_output = self.create_network(self.target_state_input, "target_Actor")
        
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Actor")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="target_Actor")

        #get target updates
        self.init_target_updates, self.soft_target_updates = self.get_target_updates()
        
        #create optimizer
        self.create_optimizer()
        
        #init variable
        self.sess.run(tf.global_variables_initializer())

        #init update target
        self.sess.run(self.init_target_updates)


    def create_network(self, state, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            fc1 = fc(state, "fc1", [self.state_dim, LAYER_SIZE1], actf=tf.nn.relu)
            fc2 = fc(fc1, "fc2", [LAYER_SIZE1, LAYER_SIZE2], actf=tf.nn.relu)
            action_output = fc(fc2, "fc3", [LAYER_SIZE2, self.action_dim], actf=tf.nn.tanh)
        return action_output
        
    def get_target_updates(self):
        init_target_updates = []
        soft_target_updates = []
        for var, target_var in zip(self.vars, self.target_vars):
            init_target_updates.append(tf.assign(target_var, var))
            soft_target_updates.append(tf.assign(target_var, (1.0-self.tau)*target_var+self.tau*var))
        return tf.group(*init_target_updates), tf.group(*soft_target_updates)

    def create_optimizer(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.vars, -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.parameters_gradients,self.vars))

    def update(self, q_gradient, state):
        self.sess.run(self.optimizer, {self.q_gradient_input:q_gradient, self.state_input:state})
        
    def action(self, state):
        return self.sess.run(self.action_output, {self.state_input:state})

    def target_action(self, state):
        return self.sess.run(self.target_action_output, {self.target_state_input:state})

if __name__=="__main__":
    import gym
    sess  = tf.Session()
    env = gym.make("Pendulum-v0")
    obs = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = Actor(sess, state_dim, action_dim, lr=0.001, tau=0.01)
    #print actor.action_output, actor.target_action_output
    #print actor.vars, actor.target_vars
    #print actor.q_gradient_input
    #print actor.parameters_gradients
    #print actor.action([obs])
    #print actor.target_action([obs])
    #actor.update([[1.0]], [obs])

