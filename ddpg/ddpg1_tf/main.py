import filter_env
from ddpg import *
import gc
gc.enable()

#ENV_NAME = 'InvertedPendulum-v1'
ENV_NAME = 'Pendulum-v0'
EPISODES = 300
TEST = 10

def main():
    env = filter_env.makeFilteredEnv(gym.make(ENV_NAME))
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)
    ave_reward = []

    for episode in xrange(EPISODES):
        state = env.reset()
        print "episode:",episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 10 == 0 and episode > 10:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_r = total_reward/TEST
                        ave_reward.append(ave_r)
			print 'episode: ',episode,'Evaluation Average Reward:',ave_r
    #env.monitor.close()
    
    import matplotlib.pyplot as plt
    plt.plot(range(len(ave_reward)), ave_reward)
    plt.show()

if __name__ == '__main__':
    main()
