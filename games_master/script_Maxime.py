import gym
import torch
import matplotlib.pyplot as plt

def parse_obs(obs):
    return torch.permute(torch.tensor(obs, dtype=torch.float), (2, 0, 1)).unsqueeze(0)

class ImageDQN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 20, 1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Flatten(),
            torch.nn.Linear(96*96*20, 5)
        )
        pass

    def forward(self, X):
        y = self.net(X)
        return y

agt = ImageDQN()
opt = torch.optim.Adam(agt.net.parameters(), lr=0.0001)
env = gym.make("CarRacing-v2", continuous=False)

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

