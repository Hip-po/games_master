import vizdoom as vzd
import numpy as np
from config import CFG
import torch
import os

def get_env():
    """
    Returns a gym environment. Replace by a custom environment if needed.
    """
    game = vzd.DoomGame()

    game.load_config(os.path.join(vzd.scenarios_path, f"{CFG.SCENARIO}.cfg"))
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, f"{CFG.SCENARIO}.wad"))
    game.set_doom_map("map01")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.set_depth_buffer_enabled(True)
    game.set_labels_buffer_enabled(True)

    game.set_render_hud(False)
    game.set_render_decals(False)
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)
    game.set_render_messages(False)
    game.set_render_corpses(False)

    game.set_window_visible(False)

    game.init()

    return game



def run_env(env, agent):
    """
    Run a given environment with a given agent.
    """

    while True:

        env.new_episode()

        state = env.get_state()

        screen_buf = torch.tensor(state.screen_buffer)
        depth_buf = torch.tensor(np.expand_dims(state.depth_buffer, axis=0))
        labels_buf = torch.tensor(np.expand_dims(state.labels_buffer, axis=0))

        new_obs=torch.cat([screen_buf,depth_buf,labels_buf])

        while not env.is_episode_finished():

            action = agent.policy(new_obs)

            old_obs=new_obs

            state = env.get_state()

            screen_buf = torch.tensor(state.screen_buffer)
            depth_buf = torch.tensor(np.expand_dims(state.depth_buffer, axis=0))
            labels_buf = torch.tensor(np.expand_dims(state.labels_buffer, axis=0))

            new_obs=torch.cat([screen_buf,depth_buf,labels_buf])

            reward = env.make_action(CFG.ACT_DICT[action])

            agent.agent_step(old_obs, action, new_obs, reward)
