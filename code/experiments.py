import numpy as np
import time
from mhn_model import ModernHopfieldNetwork
from data_generation import DataGenerator

class Experiment:
    def __init__(self, dimension=100, num_patterns=20):
        self.dimension = dimension
        self.num_patterns = num_patterns
        
    def run_single_experiment(self, beta=1.0, noise_level=0.1):
        """Один эксперимент: пробуем восстановить случайный паттерн"""
        # Генерируем паттерны
        patterns = DataGenerator.generate_patterns(self.num_patterns, self.dimension)
        
        # Создаем и настраиваем сеть Хопфилда
        mhn = ModernHopfieldNetwork(beta=beta)
        mhn.store_patterns(patterns)
        
        # Выбираем случайный паттерн и добавляем шум
        pattern_idx = np.random.randint(self.num_patterns)
        original = patterns[pattern_idx]
        query = DataGenerator.add_noise(original, noise_level)
        
        # Восстанавливаем паттерн
        start_time = time.time()
        retrieved = mhn.update(query)
        retrieval_time = time.time() - start_time
        
        # Проверяем, правильно ли восстановили
        distances = np.linalg.norm(patterns - retrieved.reshape(1, -1), axis=1)
        predicted_idx = np.argmin(distances)
        
        is_correct = (predicted_idx == pattern_idx)
        accuracy = 1.0 if is_correct else 0.0
        
        return {
            'correct': is_correct,
            'accuracy': accuracy,
            'time': retrieval_time,
            'original': original,
            'query': query,
            'retrieved': retrieved,
            'dist_to_original': np.linalg.norm(original - retrieved),
            'dist_to_query': np.linalg.norm(query - retrieved)
        }
    
    def run_statistical_experiment(self, beta=1.0, noise_level=0.1, num_trials=100):
        """Много экспериментов для статистики"""
        results = []
        
        print(f"\nЗапуск {num_trials} экспериментов:")
        print(f"  Beta: {beta}, Шум: {noise_level*100}%")
        
        for i in range(num_trials):
            result = self.run_single_experiment(beta, noise_level)
            results.append(result)
            
            # Прогресс-бар
            if (i + 1) % (num_trials // 10) == 0:
                print(f"  Прогресс: {i + 1}/{num_trials}")
        
        # Статистика
        accuracy = np.mean([r['accuracy'] for r in results])
        avg_time = np.mean([r['time'] for r in results])
        avg_dist_original = np.mean([r['dist_to_original'] for r in results])
        avg_dist_query = np.mean([r['dist_to_query'] for r in results])
        
        return {
            'accuracy': accuracy,
            'avg_time': avg_time,
            'avg_dist_original': avg_dist_original,
            'avg_dist_query': avg_dist_query,
            'results': results,
            'beta': beta,
            'noise_level': noise_level,
            'num_trials': num_trials
        }
    
    def compare_beta_values(self, noise_level=0.2, num_trials=50):
        """Сравнение разных значений beta"""
        beta_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        comparisons = []
        
        print("\n" + "="*60)
        print("СРАВНЕНИЕ РАЗНЫХ ЗНАЧЕНИЙ BETA:")
        print("="*60)
        
        for beta in beta_values:
            print(f"\nBeta = {beta}:")
            stats = self.run_statistical_experiment(beta, noise_level, num_trials)
            comparisons.append(stats)
            
            print(f"  Точность: {stats['accuracy']*100:.1f}%")
            print(f"  Среднее время: {stats['avg_time']*1000:.2f} мс")
            print(f"  Расстояние до оригинала: {stats['avg_dist_original']:.4f}")
        
        return comparisons