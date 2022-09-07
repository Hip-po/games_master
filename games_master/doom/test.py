import gym
import vizdoom as vzd
import time
import random
import os

ACT_DICT = {
    0:[False, False, False, False, False, False, False], #0 - no button
    1:[True, False, False, False, False, False, False], #1 - gauche
    2:[False, True, False, False, False, False, False], #2 - droite
    3:[False, False, True, False, False, False, False], #3 - tirer
    4:[False, False, False, True, False, False, False], #4 - avancer
    5:[False, False, False, False, True, False, False], #5 - reculer
    6:[False, False, False, False, False, True, False], #6 - rotation gauche
    7:[False, False, False, False, False, False, True], #7 - rotation droite
    8:[True, False, True, False, False, False, False], #8 - gauche + tirer
    9:[False, True, True, False, False, False, False], #9 - droite + tirer
    10:[False, False, True, True, False, False, False], #10 - avancer + tirer
    11:[False, False, True, False, True, False, False], #11 - reculer + tirer
    12:[False, False, True, False, False, True, False], #12 - rotation gauche + tirer
    13:[False, False, True, False, False, False, True] #13 - rotation droite + tirer
}

game = vzd.DoomGame()

game.load_config(os.path.join(vzd.scenarios_path, f"deadly_corridor.cfg"))
game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, f"deadly_corridor.wad"))
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

game.set_window_visible(True)

game.init()


while True:

    game.new_episode()

    while not game.is_episode_finished():

        game.make_action(ACT_DICT[random.randint(0,13)])
