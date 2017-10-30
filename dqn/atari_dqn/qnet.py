#------------------------
# File: policy network in DQN 
#       with variable scope
# Author: Jiandu Liu
# Date: 2017.10.28
#------------------------
import tensorflow as tf
import numpy as np
import gym

def max_pool_2x2(x,scope, padding="VALID"):
    with tf.variable_scope(scope):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding=padding)

def fc(x, scope, shape, actf=tf.nn.relu):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", shape, initializer = tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("b",[shape[1]],initializer = tf.constant_initializer(0.01))
        z =  actf(tf.matmul(x,w)+b)
        return z

def conv(x, scope, shape, stride, padding="VALID", actf=tf.nn.relu):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("b", [shape[3]], initializer = tf.constant_initializer(0.01))
        z =  actf(tf.nn.conv2d(x,w, strides=[1,stride, stride, 1], padding=padding)+b)
        return z


def qnet(state, nact, scope, reuse):
    with tf.variable_scope(scope, reuse=reuse):
        conv1 = conv(state,"conv1", [8,8,4,32], stride=4, padding="SAME", actf=tf.nn.relu)    
        pool1 = max_pool_2x2(conv1,"pool1", padding="SAME")
        conv2 = conv(pool1,"conv2", [4,4,32,64], stride=2, padding="SAME", actf=tf.nn.relu)
        conv3 = conv(conv2,"conv3", [3,3,64,64], stride=1, padding="SAME", actf=tf.nn.relu)
        conv3_flat = tf.reshape(conv3, [-1, 1600])
        fc4 = fc(conv3_flat,"fc4", [1600, 512],actf=tf.nn.relu)
        qvals = fc(fc4, "qval", [512, nact], actf=lambda x:x)
        return qvals



if __name__=="__main__":
    import matplotlib.pyplot as plt
    import cv2,cv

    env = gym.make("Breakout-v0")
    obs = env.reset()
    sess = tf.InteractiveSession()
   
    state = tf.placeholder(tf.float32, [None,80,80,3])
    qvals = policy_net(state, env.action_space.n,"qnet", reuse=False)
    target_qvals = policy_net(state, env.action_space.n,"qtarget_net", reuse=False)
    
    sess.run(tf.initialize_all_variables())

    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qnet")
    target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qtarget_net")
    for var, target_var in zip(vars, target_vars):
        print var.name, target_var.name
        sess.run(tf.assign(target_var, (1.0-tau)*target_var+ tau*var))
    
    
    
    obs = cv2.resize(obs, (80,80))
    label = np.array([0.1, 0.2, 0.3, 0.4])
    print sess.run(qvals, {state:[obs]})
    print sess.run(next_qvals, {state:[obs]})

