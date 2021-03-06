import numpy as np
import gym

def makeNormalizedEnv(env):
  """ crate a new environment class with actions and states normalized to [-1,1] """
  acsp = env.action_space
  obsp = env.observation_space
  if not type(acsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous action space (i.e. Box) required.')
  if not type(obsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

  env_type = type(env)

  class NormalizedEnv(env_type):
    def __init__(self):
      self.__dict__.update(env.__dict__) # transfer properties

      # Observation space
      if np.any(obsp.high < 1e10):
        h = obsp.high
        l = obsp.low
        
        sc = h-l
        self.o_c = (h+l)/2.
        self.o_sc = sc / 2.
      else:
        self.o_c = np.zeros_like(obsp.high)
        self.o_sc = np.ones_like(obsp.high)

      # Action space
      h = acsp.high
      l = acsp.low
    
      sc = (h-l)
      self.a_c = (h+l)/2.
      self.a_sc = sc / 2.

      # Rewards
      self.r_sc = 0.1
      self.r_c = 0.

      # Special cases
      '''
      if self.spec.id == "Reacher-v1":
        print "is Reacher!!!"
        self.o_sc[6] = 40.
        self.o_sc[7] = 20.
        self.r_sc = 200.
        self.r_c = 0.
    '''
      # Check and assign transformed spaces
      self.observation_space = gym.spaces.Box(self.normalize_observation(obsp.low),
                                              self.normalize_observation(obsp.high))
      self.action_space = gym.spaces.Box(-np.ones_like(acsp.high),np.ones_like(acsp.high))
      def assertEqual(a,b): assert np.all(a == b), "{} != {}".format(a,b)
      assertEqual(self.normalize_action(self.action_space.low), acsp.low)
      assertEqual(self.normalize_action(self.action_space.high), acsp.high)

    def normalize_observation(self,obs):
      return (obs-self.o_c) / self.o_sc

    def normalize_action(self,action):
      return self.a_sc*action+self.a_c

    def normalize_reward(self,reward):
      ''' has to be applied manually otherwise it makes the reward_threshold invalid '''
      return self.r_sc*reward+self.r_c

    def step(self,action):

      ac_f = np.clip(self.normalize_action(action),self.action_space.low,self.action_space.high)

      obs, reward, term, info = env_type.step(self,ac_f) # super function

      obs_f = self.normalize_observation(obs)

      return obs_f, reward, term, info

  nenv = NormalizedEnv()

  print('True action space: ' + str(acsp.low) + ', ' + str(acsp.high))
  print('True state space: ' + str(obsp.low) + ', ' + str(obsp.high))
  print('Filtered action space: ' + str(nenv.action_space.low) + ', ' + str(nenv.action_space.high))
  print('Filtered state space: ' + str(nenv.observation_space.low) + ', ' +
          str(nenv.observation_space.high))

  return nenv

if __name__=="__main__":
    import gym
    env = gym.make("Pendulum-v0")
    nenv = makeNormalizedEnv(env)
    obs = nenv.reset()
    print obs
    print env.step([0.5])
