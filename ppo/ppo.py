import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf

class Critic(object):
    def __init__(self, sess, state_dim, lr=0.01):
        self.sess = sess
        self.state_dim = state_dim
        self.lr = lr

        self.state = tf.placeholder(tf.float32, [None, self.state_dim], name="state")
        self.discounted_reward = tf.placeholder(tf.float32, [None,1],"discounted_reward")
        with tf.variable_scope("Critic"):     
            fc1 = tf.layers.dense(self.state, 100, tf.nn.relu)
            self.v_value = tf.layers.dense(fc1, 1)
            self.advantage = self.discounted_reward - self.v_value 
            self.loss = tf.reduce_mean(tf.square(self.advantage))
            self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    
    def get_advantage(self,state, discounted_reward):
        return self.sess.run(self.advantage,{self.state:state,
            self.discounted_reward:discounted_reward})
    
    def update(self,state, discounted_reward):
        return self.sess.run(self.train_step,{self.state:state,
            self.discounted_reward:discounted_reward})
    
    def get_v(self, state):
        return self.sess.run(self.v_value, {self.state:state})


class Actor(object):
    def __init__(self,sess, state_dim, action_dim, lr=0.01, **method):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.method = method

        with tf.variable_scope("Actor"):
            self.state = tf.placeholder(tf.float32, [None, self.state_dim], name="state")
            self.action = tf.placeholder(tf.float32, [None, self.action_dim], "action")
            self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")
        
        #pi and old_pi
        self.pi, self.pi_vars = self.create_pi("pi", trainable=True)
        self.sample_action = tf.squeeze(self.pi.sample(1), axis=0)

        self.old_pi, self.old_pi_vars = self.create_pi("old_pi", trainable=False)
        old_pi_update = [tf.assign(old_var, var) for var, old_var in zip(self.pi_vars, self.old_pi_vars)]
        self.old_pi_update = tf.group(*old_pi_update)

        #loss
        ratio = self.pi.prob(self.action)/self.old_pi.prob(self.action)
        surrogate = ratio*self.advantage
        if self.method["name"]=="kl_pen":
            self.lam = tf.placeholder(tf.float32, None, "lammda")
            kl = tf.contrib.distributions.kl_divergence(old_pi, pi)
            self.kl_mean = tf.reduce_mean(kl)
            self.loss = -tf.reduce_mean(surrogate-self.lam*kl)
        if self.method["name"] == "clip":
            self.loss = -tf.reduce_mean(tf.minimum(surrogate,tf.clip_by_value(ratio, 1.0-self.method["episode"],
                1.0+self.method["episode"])*self.advantage))
            
        #train step
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    

    def create_pi(self, scope, trainable):
        with tf.variable_scope(scope):
            fc1 = tf.layers.dense(self.state, 100, tf.nn.relu, trainable=trainable)
            mu = 2.0*tf.layers.dense(fc1, self.action_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(fc1, self.action_dim, tf.nn.softplus, trainable=trainable)
            pi_normal = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
            pi_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)    
        return pi_normal, pi_vars

    def choose_action(self, state):
        action = self.sess.run(self.sample_action, {self.state:state})[0]
        return np.clip(action, -2, 2)
        


class PPO(object):
    def __init__(self, 
                 env, 
                 batch_size=32,
                 actor_update_steps=10,
                 actor_lr = 0.001,
                 critic_update_steps=10,
                 critic_lr = 0.001,
                 gamma=0.99,
                 **method):

        self.sess = tf.Session()
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.method = method
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        
        self.actor_update_steps = actor_update_steps
        self.critic_update_steps = critic_update_steps
        self.gamma = gamma
        self.batch_size = batch_size

        self.critic = Critic(self.sess, self.state_dim, lr=self.critic_lr)
        self.actor = Actor(self.sess, self.state_dim, self.action_dim, lr=self.actor_lr,
                **self.method)

        self.sess.run(tf.global_variables_initializer())
        
    def update(self, state, action, discounted_reward):
        self.sess.run(self.actor.old_pi_update)
        adv = self.critic.get_advantage(state, discounted_reward)

        #update actor
        if self.method["name"] =="kl_pen":
            feed_dict={self.actor.state:state,
                       self.actor.advantage:adv,
                       self.actor.action:action,
                       self.actor.lam:self.method["lam"]}
            for _ in range(self.actor_update_steps):
                _,kl = self.sess.run([self.actor.train_step,self.actor.kl_mean], feed_dict=feed_dict)
                if kl>4*self.method["kl_target"]:
                    break
            if kl<self.method["kl_target"]/1.5:
                self.method["lam"] /=2.0
            elif kl>self.method["kl_target"]*1.5:
                self.method["lam"] *=2.0
            self.method["lam"] = np.clip(self.method["lam"], 1e-4, 10)

        elif self.method["name"]=="clip":
            feed_dict={self.actor.state:state,
                       self.actor.action:action,
                       self.actor.advantage:adv}
            for _ in range(self.actor_update_steps):
                self.sess.run(self.actor.train_step,feed_dict=feed_dict)
        
        #update critic
        for _ in range(self.critic_update_steps):
            self.critic.update(state, discounted_reward)



def main():
    env = gym.make("Pendulum-v0")
    method = [{"name":"clip", "episode":0.2},
              {"name":"kl_pen", "kl_target":0.001, "lam":0.5}][0] 

    ppo = PPO(env, 
              batch_size =32, 
              actor_rl=0.0001,
              critic_rl = 0.0002,
              gamma=0.9,
              **method)
    
    all_expected_reward = []
    max_episode = 1000
    episode_len =200


    for episode in range(max_episode):
        state = env.reset()
        state_buffer, action_buffer, reward_buffer = [], [], []
        expected_reward = 0.0
        for t in range(episode_len):
            #env.render()
            action = ppo.actor.choose_action([state])
            next_state, reward, done, _ = env.step(action)
            state_buffer.append(state)
            action_buffer.append(action)
            reward_buffer.append((reward+8.0)/8.0)
            
            state = next_state
            expected_reward+=reward

            if (t+1)%ppo.batch_size==0 or t==episode_len-1:
                v_value_next = ppo.critic.get_v([next_state])[0]
                discounted_reward_buffer=[]
                for r in reward_buffer[::-1]:
                    v_value_next = r+ppo.gamma*v_value_next
                    discounted_reward_buffer.append(v_value_next)
                discounted_reward_buffer.reverse()
                
                ppo.update(state_buffer, action_buffer, discounted_reward_buffer)

                state_buffer, action_buffer, reward_buffer = [], [], []
        
        
        if episode==0: 
            all_expected_reward.append(expected_reward)
        else: 
            all_expected_reward.append(all_expected_reward[-1]*0.9+ expected_reward*0.1)
        print ("episode ", episode, " expected_reward", expected_reward)

    print("Done")

    import matplotlib.pyplot as plt
    plt.plot(range(len(all_expected_reward)), all_expected_reward)
    plt.show()

if __name__=="__main__":
    main()
    
    ##test
    '''    
    env = gym.make("Pendulum-v0")
    sess = tf.Session()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
        
    obs = env.reset()
    print obs
    print env.step(np.array([[1.0]]))
    print env.step(np.array([1.0]))
    '''

    '''
    print env
    print sess
    print state_dim, act_dim
    print obs
    '''

    '''
    #test Critic class
    critic = Critic(sess, state_dim)
    sess.run(tf.global_variables_initializer())
    #print critic.state
    #print critic.discounted_reward
    #print critic.state_dim, critic.lr
    v= critic.get_v([obs])
    print v, 1-v

    #print critic.get_advantage([obs], [[1]])
    #critic.update([obs], [[1]])
    '''
    
    '''
    #test Actor class
    method = {"name":"clip", "episode":0.2}
    #method = dir(name="kl_pen", kl_target=0.001, lamb=0.5) 
    actor = Actor(sess, state_dim, action_dim, **method)
    sess.run(tf.global_variables_initializer())
    print actor.state_dim, actor.action_dim, actor.method
    print actor.state
    print actor.advantage
    print actor.action
    print actor.pi
    print actor.pi_vars
    print actor.old_pi
    print actor.old_pi_vars
    print actor.old_pi_update
    print actor.choose_action(obs)
    '''





