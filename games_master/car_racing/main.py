from agent import *
from environment import *
from config import CFG
from manual import manual

CFG.init()

agt = ImageDQNagent()
env = get_env()

if CFG.MANUAL:
    manual(env, agt)

run_env(env, agt)
