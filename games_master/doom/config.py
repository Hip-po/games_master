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

        self.PATH_MODEL = "model/model_doom.pt"

        self.GAMMA = 0.98
        self.EPSILON = 0.5
        self.MIN_EPSILON = 0.01
        self.ACT_RANGE = 14
        self.BATCH_SIZE = 16
        self.TARGET_FREQ = 1000
        self.SAVE_MODEL_FREQ = 5000
        self.SCENARIO="deadly_corridor"

        self.ACT_DICT = {
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


CFG = Configuration()
