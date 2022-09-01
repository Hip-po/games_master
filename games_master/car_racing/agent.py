import random
import torch
import collections
import numpy as np
from config import CFG
from save_load import save_model,load_model
from network import ImageDQN
from graph import draw_graph
import os


class ImageDQNagent():
    def __init__(self):
        self.BUFFER = collections.deque(maxlen=10000)
        if os.path.exists(CFG.PATH_MODEL):
            self.agt = load_model()
            self.tgt = load_model()
        else:
            self.agt = ImageDQN()
            self.tgt = ImageDQN()
        self.opt = torch.optim.Adam(self.agt.net.parameters(), lr=0.0001)
        self.graph=draw_graph()

    def policy(self,new_obs):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, CFG.ACT_RANGE - 1)
        with torch.no_grad():

            print(new_obs.shape)

            val = self.agt(new_obs).unsqueeze(0)

            return int(torch.argmax(val).numpy())


    def agent_step(self,old_obs, action, new_obs, reward):
        self.iter += 1

        self.BUFFER.append((old_obs, action, new_obs, reward))

        if len(self.BUFFER) >= CFG.BATCH_SIZE and self.iter % CFG.BATCH_SIZE == 0:
            # print("learn")
            # exit()
            self.learn()

        if self.iter % CFG.SAVE_MODEL_FREQ == 0:
            save_model()

        if self.iter % CFG.TARGET_FREQ == 0:
            self.tgt.load_state_dict(self.agt.state_dict())

        eps = np.exp((-self.iter - 0.15)*0.00005)
        self.epsilon = eps if eps > CFG.MIN_EPSILON else CFG.MIN_EPSILON


    def learn(self):
        batch = random.sample(CFG.BUFFER, CFG.BATCH_SIZE)
        old_obs, action, new_obs, reward = zip(*batch)

        action = torch.tensor(action).unsqueeze(1)
        reward = torch.tensor(reward)

        y_pred = torch.gather(self.agt(old_obs), 1, action).squeeze(1)

        y_true = reward + self.tgt(new_obs).max(1)[0] * CFG.GAMMA

        loss = torch.square(y_true - y_pred)

        self.graph.draw(loss, reward)

        self.opt.zero_grad()
        loss.sum().backward()
        self.opt.step()
