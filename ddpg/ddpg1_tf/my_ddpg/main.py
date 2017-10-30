import normalized_env
from ddpg import *

env_name = "InvertedPendulum-v1"
max_episode = 1000
test_episode = 10

def main()
    env = normalized_env.makeNormalziedEnv(gym.make("env_name"))
    agent = DDPG(env, 
                 actor_lr=0.01, 
                 critic_lr=0.01, 
                 tau=0.001, 
                 gamma=0.99,
                 batch_size=32, 
                 replay_memory_size=100000,
                 replay_memory_start_size=1000)

    for episode in range(max_episode):
        state = env.reset()
        #train
        for step in range(env.spec.timestep_limit):
            action = agent.noise_action(state)
            next_state, reward, done, _ = env.step(action)
            ddpg.replay_memory.add(state, action, reward, next_state, done)
            ddpg.train()
            state = next_state
            if done:
                break
        #test
        if episode%100==0:
            total_reward = 0.0
            for i in range(test_episode):
                state = env.reset()
                for j in range(env.spec.timestep_limit):
                    action = agent.actor.action(state)
                    next_state, reward, done, _ = env.step(action)
                    total_reward +=reward
                    if done:
                        break
            print ("average reward", totoal_reward/test_episode)


if __name__ == "__main__":
    main()



