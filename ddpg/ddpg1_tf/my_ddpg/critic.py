import tensorflow as tf
import numpy as np

LAYER1_SIZE=100
LAYER2_SIZE=200

def fc(x, scope, shape, actf=tf.nn.relu):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", shape=shape, initializer =
                tf.truncated_noraml_initializer(stddev=0.01))
        b = tf.get_variable("b", shape=[shape[1]],
                initialzier=tf.constant_initializer(0.01)
        z = actf(tf.matmul(x,w)+b)
    return z

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, lr, tau):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.tau = tau
        
        #create critic network and target critic network
        self.state_input = tf.placeholder("float", shape=[None, state_dim])
        self.action_input = tf.placeholder("float", shape=[None, action_dim])
        self.qval_output = create_network(self.state_input,
            self.action_input,"Critic", reuse=reuse)
        
        self.target_state_input = tf.placeholder("float", shape=[None, state_dim])
        self.target_action_input = tf.placeholder("float", shape=[None, action_dim])
        self.qval_output = create_network(self.target_state_input,
            self.target_action_input,"target_Critic", reuse=reuse)
        
        self.vars = tf.get_collection(tf.GraphKeys.TRANABLE_VARIABLE,
            scope="Critic")
        self.vars = tf.get_collection(tf.GraphKeys.TRANABLE_VARIABLE,
            scope="target_Critic")

        #get target updates
        self.init_target_upgate, self.soft_target_update = self.get_target_updates()

        #create optimizer
        self.create_optimizer()

        self.sess.run(tf.global_variables_initialzier())
        self.sess.run(self.init_target_update)

    def create_network(self, state_input, action_input, scope, reuse=reuse):
        with tf.variable_scope(scope):
            fc1 = fc(state_input, "fc1", [self.state_dim, LAYER1_SIZE], actf=tf.nn.relu)
            w2 = tf.get_variable("w2", [LAYER1_SIZE,
                    LAYER2_SIZE],initializer =
                    tf.truncated_normal_initializer(stddev=0.01))
            w2_action = tf.get_variable("w2_action", [self.action_dim,
                    LAYER2_SIZE],initializer =
                    tf.truncated_normal_initializer(stddev=0.01))
            b2 = tf.get_variable("b2", [LAYER2_SIZE+action_dim], initializer
                    = tf.constant_initialzier(0.01))
            fc2 = tf.nn.relu(tf.matmul(fc1, w2)+tf.matmul(action_input,
                    w2_action)+b2)
            qval_output = fc(fc2,"fc3", [LAYER2_SIZE, 1], actf = tf.identity)
            
        return qval_output
    
    def get_target_update(self):
        init_target_updates = []
        soft_target_updates = []
        for var, target_var in zip(self.vars, self.target_vars):
            init_target_updates.append(tf.assign(target_var, var))
            soft_target_updates.append(tf.assign(target_var,
                (1.0-self.tau)*target_var+self.tau*var))
        return tf.group(*init_target_update), tf.group(*soft_target_update)

    def create_optimizer(self):
        self.yval_input = tf.placholder("flaot", [None, 1])
        #weight_decay = tf.add_n([L2*tf.nn/l2_loss(var) for var in self.vars])
        self.loss = tf.reduce_mean(tf.square(self.qval_output-self.yval_input))
        self.optimizer =
        tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.action_gradients = tf.gradients(self.qval_output, self.action_input)

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients,
                {self.state_input:state_batch, self.action_input:action_batch})

    def update(self, yval_batch, state_batch, action_batch):
        self.sess.run(self.optimizer, {self.yval_input:yval_batch,
            self.state_input:state_batch, self.action_input:action_batch})
    
    def target_qval(self, state_batch, action_batch):
        return self.sess.run(self.target_qval_output,
                {self.state_input:state_batch, self.action_batch:action_batch})

    def qval(self, state_batch, action_batch):
        return self.sess.run(self.val_output,
                {self.state_input:state_batch, self.action_batch:action_batch})


if __name__="__main__":
    sess = tf.Session()
    env = gym.make("InvetedPendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]


    
