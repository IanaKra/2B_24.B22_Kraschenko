import matplotlib.pyplot as plt
import numpy as np

def plot_retrieval_results(original, query, retrieved):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(original.reshape(10, 10), cmap='gray')
    axes[0].set_title('Оригинальный паттерн')
    
    axes[1].imshow(query.reshape(10, 10), cmap='gray')
    axes[1].set_title('Зашумленный запрос')
    
    axes[2].imshow(retrieved.reshape(10, 10), cmap='gray')
    axes[2].set_title('Восстановленный паттерн')
    
    plt.tight_layout()
    return fig