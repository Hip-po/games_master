import gym
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation

import random
import numpy as np
import collections
from colorama import Fore, Style


import os

PATH_MODEL = "model/model_car_racing.pt"
NBR_WORKER=1
GAMMA = 0.98
EPSILON = 1
MIN_EPSILON = 0.01
ACT_RANGE = 5
BATCH_SIZE = 128
TARGET_FREQ = 1000
SAVE_MODEL_FREQ = 10000
BUFFER = collections.deque(maxlen=10000)
loss_evolution = []
frame_step = []
reward_evolution = []
NBR_WORKER=1


class ImageDQN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 20, 2),  # input_channels=3, filters = 20, kernel_size =2
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(20, 20, 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(20 * 529, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, ACT_RANGE),  # 1024 is input_dim
        )

    def forward(self, X):
        y = self.net(X)
        return y


class Worker():
    def __ini__(self, global_net, opt, name):
        print(name)
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.gnet, self.opt = global_net, opt
        self.lnet = ImageDQN()
        self.env = gym.make("CarRacing-v2", continuous=False).unwrapped
        self.agent_step_count=0

    def run(self):
        for _ in range(100000000):
            action = self.policy(new_obs)
            old_obs = new_obs
            new_obs, reward, done, info = env.step(action)
            new_obs = parse_obs(new_obs)
            self.agent_step(old_obs, action, new_obs, reward)
            if done:
                env.reset()
        env.close()

    def policy(self, new_obs):
        if random.uniform(0, 1) < EPSILON:
            return random.randint(0, ACT_RANGE - 1)
        with torch.no_grad():
            val = agt(new_obs.unsqueeze(0))
            return int(torch.argmax(val).numpy())

    def agent_step(self, old_obs, action, new_obs, reward):
        self.agent_step_count += 1

        BUFFER.append((old_obs, action, new_obs, reward))

        if len(BUFFER) >= BATCH_SIZE and self.agent_step_count % BATCH_SIZE == 0:
            self.learn()

        if self.agent_step_count % SAVE_MODEL_FREQ == 0:
            save_model()

        if self.agent_step_count % TARGET_FREQ == 0:
            tgt.load_state_dict(agt.state_dict())

        eps = np.exp(-(self.agent_step_count - 0.15))
        self.epsilon = eps if eps > MIN_EPSILON else MIN_EPSILON

    def learn(self):
        batch = random.sample(BUFFER, BATCH_SIZE)
        old_obs, action, new_obs, reward = zip(*batch)

        old_obs = torch.stack(old_obs)
        new_obs = torch.stack(new_obs)
        action = torch.tensor(action).unsqueeze(1)
        reward = torch.tensor(reward)

        y_pred = torch.gather(agt(old_obs), 1, action).squeeze(1)

        y_true = reward + tgt(new_obs).max(1)[0] * GAMMA

        loss = torch.square(y_true - y_pred)

        opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
            gp._grad = lp.grad
        opt.step()

def parse_obs(obs):
    return torch.permute(torch.tensor(obs, dtype=torch.float), (2, 0, 1))


def save_model():
    state = {
        "EPSILON": EPSILON,
        'state_dict': agt.state_dict()
    }
    torch.save(state, PATH_MODEL)
    print(Fore.BLUE + "\nSave model\n" + Style.RESET_ALL)


def load_model():
    global EPSILON
    state = torch.load(PATH_MODEL)
    model = ImageDQN()
    model.load_state_dict(state['state_dict'])
    EPSILON = state["EPSILON"]
    print(Fore.BLUE + "\nLoad model\n" + Style.RESET_ALL)
    return model


### MAIN

if os.path.exists(PATH_MODEL):
    agt = load_model()
    tgt = load_model()
else:
    agt = ImageDQN()
    tgt = ImageDQN()

opt = torch.optim.Adam(agt.net.parameters(), lr=0.0001)
env = gym.make("CarRacing-v2", continuous=False)

workers = [Worker(agt,opt,i) for i in range(NBR_WORKER)]
[w.run() for w in workers]
