from agent import *
import environment as en
from games_master.car_racing_v2.config import CFG
from manual import manual
import games_master.car_racing_v2.continuous as ct

CFG.init()

agt = ImageDQNagent()
env = en.get_env()

if CFG.MANUAL:
    manual(env, agt)

if CFG.CONTINUOUS:
    ct.run_env(env, agt)

else:
    en.run_env(env, agt)
