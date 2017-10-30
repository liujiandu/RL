import tensorflow as tf
import numpy as np

def ortho_init(scal=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == shape:
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError

        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matricess=False)
        q = u if u.shape == flat_shape else v
        q = q.reshape(shape)
    return _ortho_init



def conv(x, scope, nf, rf, stride, pad="VALID", act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[3].value
        w = tf.get_variable("w", [rf, rf, nin, nf],initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("b", [nf], initializer = tf.constant_initializer(0.0))
        z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad)+b
        h = act(z)
        return h

def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh],initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable("b", [nh], initializer = tf.constant_initializer(0.0))
        z = tf.matmul(x, w)+b
        h = act(z)
        return h

def conv_to_fc(x):
        nh = np.prod([v.value for v in x.get_shape()[1:]])
        x = tf.reshape(x, [-1, nh])
        return x

def sample(logits):
        noise = tf.random_uniform(tf.shape(logits))
        return tf.argmax(logits-tf.log(-tf.log(noise)),1)

def cat_entropy(logits):
        a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
        p0 = ea0/z0
        return tf.reduce_sum(p0*(tf.log(z0)-a0) , 1)

def find_trainable_variables(key):
    with tf.variable_scope(key):
        return tf.trainable_variables()






class Policy(object):
    def __init__(self, sess, ob_space, ac_space, nsteps, nstack, reuse):
        nh, nw, nc = ob_space.shape
        ob_shape = (nsteps, nh, nw, nc*nstack)
        nact = ac_space.n
        self.sess =sess

        self.obs = tf.placeholder(tf.uint8, ob_shape)
        with tf.variable_scope("model", reuse=reuse):
            conv1 = conv(tf.cast(self.obs, tf.float32)/255.0, "conv1", nf=32, rf=8,
                    stride=4, init_scale=np.sqrt(2))
            conv2 = conv(conv1, "conv2", nf=64, rf=4, stride=2,
                    init_scale=np.sqrt(2))
            conv3 = conv(conv2, "conv3", nf=64, rf=3, stride=1,
                    init_scale=np.sqrt(2))
            conv3_flat = conv_to_fc(conv3)
            fc4 = fc(conv3_flat, "fc4", nh=512, init_scale=np.sqrt(2))
            self.pi = fc(fc4, 'pi', nact, act=lambda x:x)
            self.vf = fc(fc4, 'v', 1, act=lambda x:x)
        self.v0 = self.vf[:,0]
        self.a0 = sample(self.pi)
        

    def step(self, ob, *_args, **_kwargs):
        a,v = self.sess.run([self.a0, self.v0], {self.obs:ob})
        return a, v, []
        
    def value(self, ob, *_args, **_kwargs):
        return self.sess.run(self.v0, {self.obs:ob})
    
    def get_pi(self, ob):
        return self.sess.run(self.pi, {self.obs:ob})


class Model(object):
    def __init__(self, policy, obs_space, ac_space, nsteps, nstack,
            ent_coef=0.01, vf_coef=0.5, total_timesteps=int(1e6),
            lrscheduler='linear', lr=7e-4, alpha=0.99, epsilon=1e-5):
        self.sess = tf.Session()
        nact = ac_space.n
       
        #step net
        self.step_net = policy(self.sess, obs_space, ac_space, 1, nstack, reuse=False)
        
        #train net
        nbatch=nsteps
        self.A = tf.placeholder(tf.int32, [nbatch])    #action 
        self.ADV = tf.placeholder(tf.float32, [nbatch]) #advantage
        self.R = tf.placeholder(tf.float32, [nbatch])    #reward+value
        self.LR = tf.placeholder(tf.float32, [])        #learning_rate
        
        self.train_net = policy(self.sess, obs_space, ac_space, nbatch, nstack,reuse=True)
        
        self.neglogpac =tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.train_net.pi,labels=self.A)
        self.pg_loss = tf.reduce_mean(self.ADV*self.neglogpac)
        self.vf_loss = tf.reduce_mean(tf.square(tf.squeeze(self.train_net.vf)-self.R)/2)
        self.entropy = tf.reduce_mean(cat_entropy(self.train_net.pi))
        self.loss = self.pg_loss - self.entropy*ent_coef + self.vf_loss* vf_coef
        
        #grad
        params = find_trainable_variables("model")
        grads = tf.gradients(self.loss, params)
        grads = list(zip(grads, params))
        
        #trainer
        self.train_step = tf.train.RMSPropOptimizer(learning_rate=self.LR, decay=alpha, epsilon=epsilon).apply_gradients(grads)
        
        #lr
        #self.lr = Scheduler(v=lr, nvalues=total_timesteps, scheduler=lrschedule)

        #init
        tf.global_variables_initializer().run(session=self.sess)
    
    def get_train_net_entropy(self, obs):
        return self.sess.run(self.entropy, {self.train_net.obs:obs})
    
    

    def train(self, obs, rewards, actions, values):
        advs = np.array(rewards) - np.array(values)
        for step in range(len(obs)):
            #cur_lr = self.lr.value()
            cur_lr = 0.001
        td_map = {self.train_net.obs:obs, self.A:actions, self.ADV:advs, self.R:rewards, self.LR:cur_lr}
        policy_loss, value_loss, policy_entropy, _ = self.sess.run([self.pg_loss, self.vf_loss, self.entropy, self.train_step], td_map)

        return policy_loss, value_loss, policy_entropy


        


class Runner(object):
        def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
            self.env =env
            self.model = model
            nh, nw, nc = env.observation_space.shape
            self.nsteps = nsteps
            nbatch=nsteps
            self.batch_ob_shape = (nbatch, nh, nw, nc*nstack)
            self.obs = np.zeros((1, nh, nw, nc*nstack), dtype=np.uint8)
            self.nc =nc
            self.gamma = gamma
            self.reset()

        def reset(self):
            obs = env.reset()
            self.update_obs(obs)
            self.done = False

        def update_obs(self, obs):
            self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
            self.obs[:,:,:,-self.nc:] = obs
        
        def run(self):
            mb_obs, mb_rewards, mb_actions ,mb_values, mb_dones = [],[],[],[],[]
        
            for n in range(self.nsteps):
                action, value, _ = self.model.step_net.step(self.obs)
                mb_obs.append(np.copy(self.obs)[0])
                mb_dones.append(self.done)
                mb_actions.append(action)
                mb_values.append(value)

                obs, reward, done, _ = self.env.step(action)
                
                mb_rewards.append(reward)

                self.done = done
                if self.done:
                    self.obs = self.obs*0
                self.update_obs(obs)
            #mb_dones.append(self.done)
            
            '''
            #batch of steps to batch of rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1,0).reshape(self.batch_ob_shape)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1,0)
            mb_actions = np.asarray(mb_actions, dtype=np.float32).swapaxes(1,0)
            mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1,0)
            mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1,0)
            '''
            #discount off value fn
            last_value = self.model.step_net.value(self.obs)[0]
            r=0
            for n in range(self.nsteps):
                r = mb_rewards[n] + self.gamma*r*(1.-mb_dones[n]) 
                if n == nsteps-1 and  mb_dones[n]==0:
                        mb_rewards[n] = r+last_value
                else:
                    mb_rewards[n] = r

            return (mb_obs, \
                    np.array(mb_rewards).flatten(), \
                    np.array(mb_actions).flatten(), \
                    np.array(mb_values).flatten(), \
                    np.array(mb_dones).flatten()) 





if __name__=="__main__":
    import gym
    env = gym.make("Breakout-v0")
    obs = env.reset()
    
    nsteps = 5
    nstack = 4
    n_episode = 10
 
    model = Model(Policy, env.observation_space, env.action_space, nsteps=nsteps,
            nstack=nstack)
    '''
    print model.step_net.value([obs])
    print model.train_net.value([obs])
    print model.get_entropy([obs])
    print model.train_net.get_pi([obs])
    '''
    
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=0.99)
    obss, rewards, actions, values, dones =  runner.run()
 
    for i in range(n_episode):
        runner.reset()
        while not runner.done:
            obss, rewards, actions, values, dones =  runner.run()
            policy_loss, value_loss, policy_entropy = model.train(obss,rewards, actions, values)
        print policy_loss
