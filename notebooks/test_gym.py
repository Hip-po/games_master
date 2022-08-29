import gym
env = gym.make("CarRacing-v2")
env.reset()

for _ in range(1000):
    action = [0,0.1,0]
    observation, reward, done, info = env.step(action)
    env.render()

    if done:
        env.reset()
env.close()
