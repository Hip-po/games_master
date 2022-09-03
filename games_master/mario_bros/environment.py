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
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        elif last_distance_max<info["distance"] :
            reward=info["distance"]-last_distance_max
            last_distance_max=info["distance"]
            pas_bouger=0
        elif last_distance==info["distance"]:
            pas_bouger+=1
            reward=0
        else:
            pas_bouger=0
            reward=0

        if pas_bouger >= 6 :
            reward=-0.01*pas_bouger
            pas_bouger+=1

        if old_score<info["score"]:
            reward+=(info["score"]/10)-old_score


        last_distance=info["distance"]


        agent.agent_step(old_obs, action, new_obs, reward)

        if done:
            last_distance_max=40
            last_distance=40
            pas_bouger=0
            old_score=0
            env.reset()

    env.close()
