import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("CarRacing-v2", n_envs=4)
env.reset()

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)

model.save("a2c_carracing.pt")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_carracing.pt")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
