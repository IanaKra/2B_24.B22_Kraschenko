import numpy as np

class DataGenerator:
    @staticmethod
    def generate_patterns(num_patterns, dimension):
        patterns = np.random.randn(num_patterns, dimension)
        patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)
        return patterns
    
    @staticmethod
    def add_noise(pattern, noise_level):
        if noise_level == 0:
            return pattern.copy()
        
        noisy = pattern.copy()
        mask = np.random.rand(*pattern.shape) < noise_level
        noise = np.random.randn(*pattern.shape)
        noisy[mask] = noise[mask]
        noisy = noisy / np.linalg.norm(noisy)
        return noisy