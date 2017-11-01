import tensorflow as tf
import numpy as np

LAYER1_SIZE=400
LAYER2_SIZE=300

def fc(x, scope, shape, actf=tf.nn.relu):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", shape=shape, initializer =
                tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("b", shape=[shape[1]],
                initializer=tf.constant_initializer(0.0))
        z = actf(tf.matmul(x,w)+b)
    return z

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, lr=0.001, tau=0.001):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.tau = tau
        
        #create critic network and target critic network
        self.state_input = tf.placeholder("float", shape=[None, state_dim])
        self.action_input = tf.placeholder("float", shape=[None, action_dim])
        self.qval_output = self.create_network(self.state_input,
            self.action_input,"Critic")
        
        self.target_state_input = tf.placeholder("float", shape=[None,state_dim])
        self.target_action_input = tf.placeholder("float", shape=[None,action_dim])
        self.target_qval_output = self.create_network(self.target_state_input,
            self.target_action_input,"target_Critic")
        
        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="Critic")
        self.target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
            scope="target_Critic")

        #get target updates
        self.init_target_updates, self.soft_target_updates = self.get_target_updates()

        #create optimizer
        self.create_optimizer()

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.init_target_updates)

    def create_network(self, state_input, action_input, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            fc1 = fc(state_input, "fc1", [self.state_dim, LAYER1_SIZE], actf=tf.nn.relu)
            w2 = tf.get_variable("w2", [LAYER1_SIZE,
                    LAYER2_SIZE],initializer =
                    tf.truncated_normal_initializer(stddev=0.01))
            w2_action = tf.get_variable("w2_action", [self.action_dim,
                    LAYER2_SIZE],initializer =
                    tf.truncated_normal_initializer(stddev=0.01))
            b2 = tf.get_variable("b2", [LAYER2_SIZE], initializer
                    = tf.constant_initializer(0.0))
            fc2 = tf.nn.relu(tf.matmul(fc1, w2)+tf.matmul(action_input,
                    w2_action)+b2)
            qval_output = fc(fc2,"fc3", [LAYER2_SIZE, 1], actf = tf.identity)
            
        return qval_output
    
    def get_target_updates(self):
        init_target_updates = []
        soft_target_updates = []
        for var, target_var in zip(self.vars, self.target_vars):
            init_target_updates.append(tf.assign(target_var, var))
            soft_target_updates.append(tf.assign(target_var,
                (1.0-self.tau)*target_var+self.tau*var))
        return tf.group(*init_target_updates), tf.group(*soft_target_updates)

    def create_optimizer(self):
        self.yval_input = tf.placeholder("float", [None, 1])
        #weight_decay = tf.add_n([L2*tf.nn/l2_loss(var) for var in self.vars])
        self.loss = tf.reduce_mean(tf.square(self.qval_output-self.yval_input))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.action_gradients = tf.gradients(self.qval_output, self.action_input)

    def gradients(self, state, action):
        return self.sess.run(self.action_gradients,
                {self.state_input:state, self.action_input:action})

    def update(self, yval, state, action):
        self.sess.run(self.optimizer, {self.yval_input:yval,
            self.state_input:state, self.action_input:action})
    
    def target_qval(self, state, action):
        return self.sess.run(self.target_qval_output,
                {self.target_state_input:state, self.target_action_input:action})

    def qval(self, state, action):
        return self.sess.run(self.qval_output,
                {self.state_input:state, self.action_input:action})


if __name__=="__main__":
    import gym
    sess = tf.Session()
    env = gym.make("Pendulum-v0")
    obs = env.reset()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    #print action_dim
    #print state_dim
    #print obs

    critic =Critic(sess, state_dim, action_dim, lr=0.001,tau=0.001)
    #print critic.qval_output, critic.target_qval_output
    #print critic.vars, critic.target_vars
    #print critic.init_target_updates, critic.soft_target_updates
    
    #print critic.loss
    #print critic.optimizer
    #print critic.action_gradients
    #print critic.qval([obs,obs], [[1],[1]])
    #print critic.target_qval([obs,obs], [[1],[1]])
    #print critic.gradients([obs,obs], [[1], [1]])
    #critic.update([[1]], [obs], [[1]])
