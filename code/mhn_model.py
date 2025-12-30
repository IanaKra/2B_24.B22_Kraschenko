import numpy as np

class ModernHopfieldNetwork:
    def __init__(self, beta=1.0):
        self.beta = beta
        self.K = None  # ключи
        self.V = None  # значения
        
    def store_patterns(self, patterns):
        """patterns: матрица (m, d) - m паттернов по d нейронов"""
        self.K = patterns.T  # (d, m)
        self.V = patterns.T  # (d, m)
        
    def update(self, query):
        """Один шаг обновления MHN = softmax attention"""
        # query: вектор (d,)
        # self.K: (d, m)
        # self.V: (d, m)
        
        # Вычисляем оценки
        scores = self.beta * np.dot(self.K.T, query)  # (m,)
        
        # Стабильный softmax
        scores = scores - np.max(scores)  # для численной стабильности
        exp_scores = np.exp(scores)
        weights = exp_scores / np.sum(exp_scores)  # (m,)
        
        # Взвешенная сумма значений
        output = np.dot(self.V, weights)  # (d,)
        
        return output
    
    def energy(self, x):
        """Энергия MHN: -1/beta * log(sum(exp(beta * K^T x))) + 0.5 * ||x||^2"""
        # x: вектор состояния (d,)
        # self.K: (d, m)
        
        quadratic = 0.5 * np.dot(x, x)
        exp_terms = np.exp(self.beta * np.dot(self.K.T, x))  # (m,)
        log_sum = np.log(np.sum(exp_terms))
        
        energy = -log_sum / self.beta + quadratic
        return energy