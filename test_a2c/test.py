import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
GAMMA = 0.98
EPSILON = 1
MIN_EPSILON = 0.01
ACT_RANGE = 5
BATCH_SIZE = 128
TARGET_FREQ = 1000
SAVE_MODEL_FREQ = 10000
reward_evolution = []

env = make_vec_env("CarRacing-v0", n_envs=4)
######################################
#version qui marche

model = A2C("MlpPolicy", env, verbose=1,
           learning_rate=0.0001,
           gamma = GAMMA,
           n_steps=BATCH_SIZE,
           #create_eval_env=True
           )
model.learn(total_timesteps=20000
            )

model.save("a2c_carracing.pt")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_carracing.pt")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()



#######################################
#testing with previous version
new_obs = env.reset()

for i in range(100_000):
    action, _states = model.predict(new_obs)
    olb_obs = new_obs
    new_obs, reward, done, info = env.step(action)
    #model.learn(total_timesteps=20000)
    #model.save("a2c_carracing.pt")
    reward_evolution.append(reward)
    print(reward)
    #agent_step(old_obs, action, new_obs, reward)
    env.render()
