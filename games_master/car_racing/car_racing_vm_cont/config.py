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

        self.MANUAL = False
        self.GRAYSCALE = False
        self.CONTINUOUS = True
        self.VM=True
        self.GRAPH=False

        self.PATH_MODEL = f"""model/model_car_racing_{"discret" if not self.CONTINUOUS else "cont"}_{'preprocessed' if self.GRAYSCALE else 'RAW'}_v2fixedLR.pt"""

        self.GAMMA = 0.98
        self.EPSILON = 0.5
        self.MIN_EPSILON = 0.01
        self.ACT_RANGE = 5 if not CFG.CONTINUOUS else 2000
        self.BATCH_SIZE = 128
        self.TARGET_FREQ = 1000
        self.SAVE_MODEL_FREQ = 10000

        self.dict_choice = {
            ' ': 0,
            'q': 1,
            'd': 2,
            'z': 3,
            's': 4
        }


CFG = Configuration()
