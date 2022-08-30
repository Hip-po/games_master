"""
Environment module.
This module contains the RL environment. We provide a gym setup by default, which can easily be replaced by other packages such as pettingzoo. Fundamentally, this module is used to simulate the environment and generate (s, a, r, s') tuples for the agent to learn from.
"""

import gym
import torch

from config import CFG


def parse_obs(obs):
    return torch.permute(torch.tensor(obs, dtype=torch.float), (2, 0, 1)).unsqueeze(0)

def get_env():
    """
    Returns a gym environment. Replace by a custom environment if needed.
    """
    # We use the LunarLander env. Other environments are available.
    return gym.make("CarRacing-v2", continuous=False)


def run_env(env, agt):
    """
    Run a given environment with a given agent.
    """

    obs_old, info = env.reset(seed=CFG.rnd_seed, return_info=True)

    opt = torch.optim.Adam(agt.net.parameters(), lr=0.0001)
    # We get the action space.
    act_space = env.action_space

    new_obs = parse_obs(env.reset())

    for _ in range(10000):

        val = agt(new_obs)
        act = torch.argmax(val).numpy()
        old_obs = new_obs
        print(val, act)

        new_obs, reward, done, info = env.step(act)
        new_obs = parse_obs(new_obs)

        env.render()

        out = val.squeeze(0)[act]
        with torch.no_grad():
            exp = reward + 0.98 * agt(new_obs).max()

        loss = torch.square(exp - out)

        opt.zero_grad()
        loss.sum().backward()
        opt.step()

        if done:
            env.reset()
    env.close()
