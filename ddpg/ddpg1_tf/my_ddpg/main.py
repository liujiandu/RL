import normalized_env
from ddpg import *

env_name = "Pendulum-v0"
max_episode = 300
test_episode = 10

def main():
    env = normalized_env.makeNormalizedEnv(gym.make(env_name))
    ddpg = DDPG(env, 
                actor_lr=0.0001, 
                critic_lr=0.001, 
                tau=0.001, 
                gamma=0.99,
                batch_size=32, 
                replay_memory_size=100000,
                replay_memory_start_size=10000)
    
    ave_reward = []
    for episode in range(max_episode):
        print("episode ", episode)
        state = env.reset()
        
        #train
        for step in range(env.spec.timestep_limit):
            action = ddpg.noise_action([state])[0]
            next_state, reward, done, _ = env.step(action)
            ddpg.replay_memory.add(state, action, reward, next_state, done)    
            
            if ddpg.replay_memory.count() > ddpg.replay_memory_start_size:
                ddpg.train()
            
            state = next_state
            if done:
                break
        #test
        if (episode)%10==0 and episode>0:
            total_reward = 0.0
            for i in range(test_episode):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    action = ddpg.actor.action([state])[0]
                    state, reward, done, _ = env.step(action)
                    total_reward +=reward
                    if done:
                        break

            ave_r = total_reward/test_episode
            ave_reward.append(ave_r)
            print ("average reward", ave_r)

    import matplotlib.pyplot as plt
    plt.plot(range(len(ave_reward)), ave_reward)
    plt.show()

if __name__ == "__main__":
    main()



