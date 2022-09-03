import random


class Configuration:
    """
    This configuration class is extremely flexible due to a two-step init process. We only instanciate a single instance of it (at the bottom if this file) so that all modules can import this singleton at load time. The second initialization (which happens in main.py) allows the user to input custom parameters of the config class at execution time.
    """

    def __init__(self):
        """
        Declare types but do not instanciate anything
        """
        pass

    def init(self):
        """
        User-defined configuration init. Mandatory to properly set all configuration parameters.
        """

        self.PATH_MODEL = "model/model_mario_bross_v1.pt"

        self.GAMMA = 0.98
        self.EPSILON = 0.5
        self.MIN_EPSILON = 0.01
        self.ACT_RANGE = 11
        self.BATCH_SIZE = 128
        self.TARGET_FREQ = 1000
        self.SAVE_MODEL_FREQ = 10000

        self.ACT_DICT = {
            0: [0, 0, 0, 0, 0, 0],  # No action
            1: [1, 0, 0, 0, 0, 0],  # Up
            2: [0, 1, 0, 0, 0, 0],  # Left
            3: [0, 0, 1, 0, 0, 0],  # Down
            4: [0, 0, 0, 1, 0, 0],  # Right
            5: [0, 0, 0, 0, 1, 0],  # A
            6: [0, 0, 0, 0, 0, 1],  # B
            7: [0, 1, 0, 0, 1, 0],  # Jump + Left
            8: [0, 0, 0, 1, 1, 0],  # Jump + Right
            9: [0, 1, 0, 0, 0, 1],  # Run + Left
            10: [0, 0, 0, 1, 0, 1],  # Run + Right
        }


CFG = Configuration()
