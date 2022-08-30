import gym
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import collections

GAMMA = 0.98
EPSILON = 0.1
ACT_RANGE = 5
BATCH_SIZE = 128
TARGET_FREQ = 1000
BUFFER = collections.deque(maxlen=100000)

class ImageDQN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(20, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(20*529, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 5),
        )

    def forward(self, X):
        y = self.net(X)
        return y

def parse_obs(obs):
    return torch.permute(torch.tensor(obs, dtype=torch.float), (2, 0, 1))

def policy(new_obs):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, ACT_RANGE - 1)
    with torch.no_grad():
        val = agt(new_obs.unsqueeze(0))
        return int(torch.argmax(val).numpy())

def agent_step(old_obs, action, new_obs, reward):

    agent_step.iter += 1

    BUFFER.append((old_obs, action, new_obs, reward))

    if len(BUFFER) >= BATCH_SIZE and agent_step.iter % BATCH_SIZE == 0:
        learn()

    if agent_step.iter % TARGET_FREQ == 0:
        tgt.load_state_dict(agt.state_dict())

if "iter" not in agent_step.__dict__:
    agent_step.iter = 0

def learn():

    batch = random.sample(BUFFER, BATCH_SIZE)
    old_obs, action, new_obs, reward = zip(*batch)

    old_obs = torch.stack(old_obs)
    new_obs = torch.stack(new_obs)
    action = torch.tensor(action).unsqueeze(1)
    reward = torch.tensor(reward)

    y_pred = torch.gather(agt(old_obs), 1, action).squeeze(1)

    # with torch.no_grad():
    #     y_true = reward + tgt(new_obs).max(1)[0] * GAMMA

    y_true = reward + tgt(new_obs).max(1)[0] * GAMMA

    loss = torch.square(y_true - y_pred)
    print(loss.sum())

    opt.zero_grad()
    loss.sum().backward()
    opt.step()

### MAIN

agt = ImageDQN()
tgt = ImageDQN()

for param in tgt.parameters():
    param.requires_grad = False

opt = torch.optim.Adam(agt.net.parameters(), lr=0.0001)
env = gym.make("CarRacing-v2", continuous=False)

new_obs = parse_obs(env.reset())

for i in range(10000000):

    action = policy(new_obs)

    old_obs = new_obs

    new_obs, reward, done, info = env.step(action)
    new_obs = parse_obs(new_obs)
    agent_step(old_obs, action, new_obs, reward)

    env.render()

    if done:
        env.reset()
env.close()
