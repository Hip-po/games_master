from config import CFG
import gym
import numpy as np
import torch


#defining the actions
steer = np.linspace(-1.0, 1.0, num=20)
gas = np.linspace(0, 1.0, num=10)
brake = np.linspace(0, 1.0, num=10)
cont_action = [[s, g, b] for s in steer for g in gas for b in brake]

def parse_obs(obs):
    return torch.permute(torch.tensor(obs, dtype=torch.float), (2, 0, 1))

def run_env(env, agent):
    """
    Run a given environment with a given agent.
    """

    new_obs = env.reset()

    while True:

        action = agent.policy(new_obs)

        old_obs = new_obs

        new_obs, reward, done, _ = env.step(cont_action[action])
        #new_obs = parse_obs(new_obs)
        agent.agent_step(old_obs, action, new_obs, reward)

        #env.render()

        if done:
            env.reset()

    env.close()
