import gym
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation

import random
import numpy as np
import collections
from colorama import Fore, Style

GAMMA = 0.98
EPSILON=1
MIN_EPSILON = 0.01
ACT_RANGE = 5
BATCH_SIZE = 128
TARGET_FREQ = 1000
SAVE_MODEL_FREQ=10000
BUFFER = collections.deque(maxlen=10000)
loss_evolution = []
frame_step = []
reward_evolution = []


plt.ion()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
line, = ax1.plot([], [])
ax1.set_xlabel('batch #')
ax1.set_ylabel('Loss')
ax1.set_title('Loss evolution')
line2, = ax2.plot([], [])
ax2.set_xlabel('batch #')
ax2.set_ylabel('Reward')
ax2.set_title('Reward evolution')
plt.tight_layout()

class ImageDQN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 20, 2),  #input_channels=3, filters = 20, kernel_size =2
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
            torch.nn.Linear(1024, ACT_RANGE), # 1024 is input_dim
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

    if agent_step.iter % SAVE_MODEL_FREQ==0:
        save_model()

    if agent_step.iter % TARGET_FREQ == 0:
        tgt.load_state_dict(agt.state_dict())

    eps=np.exp(-(agent_step.iter-0.15))
    EPSILON = eps if eps > MIN_EPSILON else MIN_EPSILON

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

    y_true = reward + tgt(new_obs).max(1)[0] * GAMMA

    loss = torch.square(y_true - y_pred)
    #print(loss)
    loss_evolution.append(float(loss.sum().detach().numpy()))
    reward_evolution.append(float(reward.sum().detach().numpy()))
    frame_step.append(agent_step.iter)
    print(reward_evolution)
    # print(loss_evolution)
    # print(list(range(len(loss_evolution))))
    # print(len(loss_evolution))
    # print(min(loss_evolution[-20:]))
    #LOSS
    if len(loss_evolution)<20:
        ax1.set_ylim(0, 100000)
        ax1.set_xlim(0, 100)
    else:
        ax1.set_xlim(len(loss_evolution)-20, len(loss_evolution))
        ax1.set_ylim(min(loss_evolution[-20:]),max(loss_evolution[-20:]))
    line.set_xdata(list(range(len(loss_evolution))))
    line.set_ydata(loss_evolution)
    line.figure.canvas.draw_idle()

    #REWARD
    if len(reward_evolution)<20:
        ax2.set_ylim(-50, 50)
        ax2.set_xlim(0, 100)
    else:
        ax2.set_xlim(len(reward_evolution)-20, len(reward_evolution))
        ax2.set_ylim(min(reward_evolution[-20:]),max(reward_evolution[-20:]))
    line2.set_xdata(list(range(len(reward_evolution))))
    line2.set_ydata(reward_evolution)
    line2.figure.canvas.draw_idle()

    opt.zero_grad()
    loss.sum().backward()
    opt.step()

def save_model():
    print(Fore.BLUE + "\nSave model\n" + Style.RESET_ALL)
    torch.save(agt.state_dict(), "model/model_car_racing.pt")

def load_model():
    print(Fore.BLUE + "\nLoad model\n" + Style.RESET_ALL)
    model = ImageDQN()
    model.load_state_dict(torch.load("model/model_car_racing.pt"), strict=False)
    return model

### MAIN

try:
    agt = load_model()
    tgt = load_model()
except:
    agt = ImageDQN()
    tgt = ImageDQN()

# agt = load_model()
# tgt = load_model()

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