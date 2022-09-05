import gym
import ppaquette_gym_super_mario
import time
import random
import matplotlib.pyplot as plt


env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

state=env.reset()

while True:
    act=[0, 0, 0, 0, 0, 0]
    udx=random.randint(0,5)
    if udx != 1 :
        act[udx]=1

    state, reward, done, info = env.step(act)
    # if index % 25 ==0:
    #     print(state)
    #     plt.imshow(state)
    #     plt.show()
    time.sleep(0.05)

    if done:
        env.reset()

env.close()
