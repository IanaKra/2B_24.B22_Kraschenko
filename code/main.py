import numpy as np
import matplotlib.pyplot as plt
from experiments import Experiment

def main():
    print("=" * 60)
    print("ЭКСПЕРИМЕНТ: СОВРЕМЕННАЯ СЕТЬ ХОПФИЛДА")
    print("Сравнение разных значений параметра Beta")
    print("=" * 60)
    
    # Создаем эксперимент
    exp = Experiment(dimension=100, num_patterns=20)
    
    # Запускаем сравнение разных beta
    comparisons = exp.compare_beta_values(noise_level=0.2, num_trials=10)
    
    # Выводим сводную таблицу
    print("\n" + "="*60)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ:")
    print("="*60)
    print(f"{'Beta':<8} {'Точность, %':<15} {'Время, мс':<12} {'Расстояние':<12}")
    print("-"*50)
    
    for comp in comparisons:
        print(f"{comp['beta']:<8} {comp['accuracy']*100:<15.1f} {comp['avg_time']*1000:<12.2f} {comp['avg_dist_original']:<12.4f}")
    
    # Визуализация: точность в зависимости от beta
    betas = [comp['beta'] for comp in comparisons]
    accuracies = [comp['accuracy'] for comp in comparisons]
    
    plt.figure(figsize=(10, 6))
    plt.plot(betas, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Beta (обратная температура)', fontsize=12)
    plt.ylabel('Точность восстановления', fontsize=12)
    plt.title('Зависимость точности MHN от параметра Beta\n(шум 20%, 10 испытаний на точку)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(betas)
    plt.ylim(0, 1.0)
    
    # Сохраняем график
    plt.savefig('beta_comparison.png', dpi=150, bbox_inches='tight')
    print("\nГрафик сохранен как 'beta_comparison.png'")
    
    # Демонстрация одного примера восстановления
    print("\n" + "="*60)
    print("ПРИМЕР ВОССТАНОВЛЕНИЯ ПАТТЕРНА (Beta=2.0):")
    print("="*60)
    
    # Запускаем один эксперимент для демонстрации
    single_result = exp.run_single_experiment(beta=2.0, noise_level=0.2)
    
    print(f"Результат: {'УСПЕХ' if single_result['correct'] else 'ОШИБКА'}")
    print(f"Время восстановления: {single_result['time']*1000:.2f} мс")
    print(f"Расстояние до оригинала: {single_result['dist_to_original']:.4f}")
    print(f"Расстояние до запроса: {single_result['dist_to_query']:.4f}")
    
    # Создаем график для этого примера
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Преобразуем векторы в изображения 10x10
    original_img = single_result['original'].reshape(10, 10)
    query_img = single_result['query'].reshape(10, 10)
    retrieved_img = single_result['retrieved'].reshape(10, 10)
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Оригинальный паттерн')
    axes[0].axis('off')
    
    axes[1].imshow(query_img, cmap='gray')
    axes[1].set_title('Зашумленный запрос (20% шума)')
    axes[1].axis('off')
    
    axes[2].imshow(retrieved_img, cmap='gray')
    axes[2].set_title('Восстановленный паттерн')
    axes[2].axis('off')
    
    plt.savefig('single_example.png', dpi=150, bbox_inches='tight')
    print("Пример сохранен как 'single_example.png'")
    
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ ЗАВЕРШЕН!")
    print("="*60)

if __name__ == "__main__":
    main()