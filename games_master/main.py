import agent
import environment

from config import CFG
from script_Maxime import ImageDQN

# We initialize our configuration class
CFG.init("", rnd_seed=22)

# We create an agent. State and action spaces are hardcoded here.
agt = ImageDQN()

# Run a learning process
for _ in range(1000):
    env = environment.get_env()
    environment.run_env(env, agt)
