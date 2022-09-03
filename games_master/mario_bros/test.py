import gym
import ppaquette_gym_super_mario
import time
import random
import matplotlib.pyplot as plt


env = gym.make('ppaquette/SuperMarioBros-1-1-Tiles-v0')

state=env.reset()
index=0
while True:
    index += 1
    if index % 2==0:
        act=[0, 0, 0, 1, 1, 0]
    else:
        act=[0, 0, 0, 1, 0, 0]
    # udx=random.randint(0,5)
    # if udx != 1 :
    #     act[udx]=1

    state, reward, done, info = env.step(act)
    # if index % 25 ==0:
    #     print(state)
    #     plt.imshow(state)
    #     plt.show()
    time.sleep(0.02)

    if done:
        env.reset()

env.close()
