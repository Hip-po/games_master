from agent import *
from environment import *
from config import CFG

CFG.init()

agt = ImageDQNagent()
env = get_env()

run_env(env, agt)
