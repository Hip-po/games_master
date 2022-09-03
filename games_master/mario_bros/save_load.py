import torch
from config import CFG
from colorama import Fore, Style
from network import ImageDQN


def save_model(agt, EPSILON):
    state = {
        "EPSILON": EPSILON,
        'state_dict': agt.state_dict()
    }
    torch.save(state, CFG.PATH_MODEL)
    print(Fore.BLUE + "\nSave model\n" + Style.RESET_ALL)


def load_model():
    global EPSILON
    state = torch.load(CFG.PATH_MODEL)
    model = ImageDQN()
    model.load_state_dict(state['state_dict'])
    EPSILON = state["EPSILON"]
    print(Fore.BLUE + "\nLoad model\n" + Style.RESET_ALL)
    return model, EPSILON
