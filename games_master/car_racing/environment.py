import gym
from config import CFG

def get_env():
    """
    Returns a gym environment. Replace by a custom environment if needed.
    """
    # We use the LunarLander env. Other environments are available.
    return gym.make("CarRacing-v2", continuous=CFG.CONTINUOUS)


def run_env(env, agent):
    """
    Run a given environment with a given agent.
    """

    new_obs = env.reset()

    while True:

        action = agent.policy(new_obs)

        old_obs = new_obs

        new_obs, reward, done, _ = env.step(action)
        agent.agent_step(old_obs, action, new_obs, reward)

        env.render()

        if done:
            env.reset()

    env.close()
