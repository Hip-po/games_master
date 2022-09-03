from agent import *
from environment import *
import ppaquette_gym_super_mario
from config import CFG

CFG.init()

agt = ImageDQNagent()
env = get_env()

run_env(env, agt)
