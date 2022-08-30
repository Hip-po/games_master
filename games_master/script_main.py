import gym
import torch
import matplotlib.pyplot as plt
import random

GAMMA = 0.98
EPSILON = 0.1
ACT_RANGE = 5

def parse_obs(obs):
    return torch.permute(torch.tensor(obs, dtype=torch.float), (2, 0, 1)).unsqueeze(0)

def policy(new_obs):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, ACT_RANGE - 1)
    with torch.no_grad():
        return torch.argmax(agt(new_obs)).numpy()

def learn(old_obs, action, new_obs, reward):

    out = agt(old_obs).squeeze(0)[action]
    with torch.no_grad():
        exp = reward + GAMMA * agt(new_obs).max()

    loss = torch.square(exp - out)
    print(action, reward, loss)

    opt.zero_grad()
    loss.sum().backward()
    opt.step()

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
            torch.nn.Flatten(start_dim=0),
            torch.nn.Linear(20*529, 1024),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(1024, 5),
        )

    def forward(self, X):
        y = self.net(X)
        return y

agt = ImageDQN()
opt = torch.optim.Adam(agt.net.parameters(), lr=0.0001)
env = gym.make("CarRacing-v2", continuous=False)

new_obs = parse_obs(env.reset())

for _ in range(10000):

    action = policy(new_obs)

    old_obs = new_obs

    new_obs, reward, done, info = env.step(action)
    new_obs = parse_obs(new_obs)
    learn(old_obs, action, new_obs, reward)

    env.render()

    if done:
        env.reset()
env.close()
