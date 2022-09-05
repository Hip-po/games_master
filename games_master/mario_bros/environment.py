from cmath import inf
from distutils.log import info
import gym
from config import CFG
import numpy as np
import time

def get_env():
    """
    Returns a gym environment. Replace by a custom environment if needed.
    """
    # We use the LunarLander env. Other environments are available.
    return gym.make("ppaquette/SuperMarioBros-1-1-Tiles-v0")


def run_env(env, agent):
    """
    Run a given environment with a given agent.
    """

    new_obs = env.reset()

    last_distance_max=40
    last_distance=40
    pas_bouger=0
    old_score=0


    while True:

        action = agent.policy(new_obs)

        old_obs = new_obs

        new_obs, _, done, info = env.step(CFG.ACT_DICT[action])

        #calcul reward

        if done :
            if "flag_get" in info:
                reward= 200
            elif info["time"]==0:
                reward= 0
            else:
                reward = -25
        elif last_distance_max<int(info["distance"]) :
            reward=int(info["distance"])-last_distance_max
            last_distance_max=int(info["distance"])
            pas_bouger=0
        elif last_distance==int(info["distance"]):
            if pas_bouger>=5:
                reward=-0.005*pas_bouger
            else:
                reward=0
            pas_bouger+=1
        else:
            pas_bouger=0
            reward=0

        if old_score<info["score"]/10:
            reward+=(info["score"]/10)-old_score
            old_score=info["score"]/10

        last_distance=info["distance"]

        # print(reward)
        # time.sleep(0.05)

        agent.agent_step(old_obs, action, new_obs, reward)

        if done:
            last_distance_max=40
            last_distance=40
            pas_bouger=0
            old_score=0
            env.reset()

    env.close()
