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
        self.ACT_RANGE = 14
        self.BATCH_SIZE = 128
        self.TARGET_FREQ = 1000
        self.SAVE_MODEL_FREQ = 10000

        self.ACT_DICT = {
            0:[0, 0, 0, 0, 0, 0], #0 - no button,\n",
            1:[1, 0, 0, 0, 0, 0], #1 - up only (to climb vine)\n",
            2:[0, 1, 0, 0, 0, 0], #2 - left only\n",
            3:[0, 0, 1, 0, 0, 0], #3 - down only (duck, down pipe)\n",
            4:[0, 0, 0, 1, 0, 0], #4 - right only\n",
            5:[0, 0, 0, 0, 0, 1], #5 - run only\n",
            6:[0, 0, 0, 0, 1, 0], #6 - jump only\n",
            7:[0, 1, 0, 0, 0, 1], #7 - left run\n",
            8:[0, 1, 0, 0, 1, 0], #8 - left jump\n",
            9:[0, 0, 0, 1, 0, 1], #9 - right run\n",
            10:[0, 0, 0, 1, 1, 0], #10 - right jump\n",
            11:[0, 1, 0, 0, 1, 1], #11 - left run jump\n",
            12:[0, 0, 0, 1, 1, 1], #12 - right run jump\n",
            13:[0, 1, 0, 0, 1, 0] #13 - down jump\n",
        }


CFG = Configuration()
