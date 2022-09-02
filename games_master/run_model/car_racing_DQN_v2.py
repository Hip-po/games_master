import gym
import torch
from colorama import Fore, Style
import torchvision.transforms as tv
import numpy as np
import time


def run():

    PATH_MODEL = "model/model_car_racing_v2.pt"
    ONE_TRACK=True
    TOTAL_RWD=0
    GRAYSCALE = True

    class ImageDQN(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(1 if GRAYSCALE else 3, 20, 2),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(20, 20, 2),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(20*483, 1024),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(1024, 5),
            )
            self.img = tv.transforms.Compose([tv.transforms.ToTensor(),
                                            tv.transforms.Grayscale(),
                                            tv.Lambda(lambda x: tv.functional.crop(x, 0, 0, 88, 96))
                                            ])

        def forward(self, X):
            X = torch.stack([self.img(x) for x in X])
            y = self.net(X)
            return y


    def policy(new_obs):
        with torch.no_grad():
            val = agt(np.expand_dims(new_obs, 0))
            return int(torch.argmax(val).numpy())

    def load_model():
        global EPSILON
        state = torch.load(PATH_MODEL)
        model = ImageDQN()
        model.load_state_dict(state['state_dict'])
        EPSILON = state["EPSILON"]
        print(Fore.BLUE + "\nLoad model\n" + Style.RESET_ALL)
        return model, EPSILON

    agt, _ = load_model()

    env = gym.make("CarRacing-v2", continuous=False)

    new_obs = env.reset()

    while True:

        action = policy(new_obs)

        _, reward, done, _ = env.step(action)

        TOTAL_RWD+=reward

        env.render()

        if done:
            env.reset()
            if ONE_TRACK:
                break

    return TOTAL_RWD

if __name__=="__main__":
    run()
