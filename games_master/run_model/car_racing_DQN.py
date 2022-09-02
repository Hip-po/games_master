import gym
import torch
from colorama import Fore, Style


def run():

    PATH_MODEL = "model/model_car_racing_QDN.pt"
    ONE_TRACK=True
    TOTAL_RWD=0

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
        with torch.no_grad():
            val = agt(new_obs.unsqueeze(0))
            return int(torch.argmax(val).numpy())

    def load_model():
        state = torch.load(PATH_MODEL)
        model = ImageDQN()
        model.load_state_dict(state['state_dict'])
        print(Fore.BLUE + "\nLoad model\n" + Style.RESET_ALL)
        return model

    agt = load_model()

    env = gym.make("CarRacing-v2", continuous=False)
    new_obs = parse_obs(env.reset())

    while True:

        action = policy(new_obs)

        new_obs, reward, done, _ = env.step(action)
        new_obs = parse_obs(new_obs)

        TOTAL_RWD+=reward

        env.render()

        if done:
            env.reset()
            if ONE_TRACK:
                break
    return TOTAL_RWD

if __name__=="__main__":
    run()
