import numpy as np

class Config:
    SEED = 42
    np.random.seed(SEED)
    
    DIMENSION = 100
    NUM_PATTERNS = 20
    BETA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
    NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    NUM_TRIALS = 50
    MAX_ITERATIONS = 50
    LEARNING_RATE = 0.1
    
    SAVE_DIR = "results/"
    FIGURES_DIR = "figures/"