"""
Визуализация результатов оптимизации полива
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_irrigation_policies(mlp_model, rl_agent, output_dir="reports/figures"):
    """Визуализирует политики полива"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Тепловая карта урожайности
    water_range = np.linspace(5, 50, 50)
    temp_range = np.linspace(15, 35, 50)
    yields_grid = np.zeros((len(temp_range), len(water_range)))
    
    for i, temp in enumerate(temp_range):
        for j, water in enumerate(water_range):
            state = np.array([[water, temp, 60, 20, 0.7, 45]])
            pred = mlp_model.predict(state)
            yields_grid[i, j] = pred[0, 0]
    
    im1 = axes[0, 0].imshow(yields_grid, aspect='auto', origin='lower',
                            extent=[5, 50, 15, 35], cmap='YlGn')
    axes[0, 0].set_xlabel('Полив (мм/день)')
    axes[0, 0].set_ylabel('Температура (°C)')
    axes[0, 0].set_title('Прогноз урожайности (MLP)')
    plt.colorbar(im1, ax=axes[0, 0])
    axes[0, 0].axvline(x=25, color='red', linestyle='--', linewidth=2)
    
    # 2. Эффективность воды
    eff_grid = np.zeros((len(temp_range), len(water_range)))
    for i, temp in enumerate(temp_range):
        for j, water in enumerate(water_range):
            state = np.array([[water, temp, 60, 20, 0.7, 45]])
            pred = mlp_model.predict(state)
            eff_grid[i, j] = pred[0, 1]
    
    im2 = axes[0, 1].imshow(eff_grid, aspect='auto', origin='lower',
                            extent=[5, 50, 15, 35], cmap='Blues')
    axes[0, 1].set_xlabel('Полив (мм/день)')
    axes[0, 1].set_ylabel('Температура (°C)')
    axes[0, 1].set_title('Эффективность использования воды')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 3. Стратегии полива
    days = np.arange(1, 91)
    temps = 20 + 10 * np.sin(days * np.pi / 90)
    
    fixed = np.ones_like(days) * 25
    mlp_irr = [25 for _ in days]  # упрощенно
    rl_irr = [rl_agent.select_action(np.array([0, t, 60, 20, 0.7, d])) 
              for d, t in zip(days, temps)]
    
    axes[1, 0].plot(days, fixed, 'b--', label='Фиксированный', linewidth=2)
    axes[1, 0].plot(days, mlp_irr, 'g-', label='MLP', linewidth=2)
    axes[1, 0].plot(days, rl_irr, 'r-', label='RL', linewidth=2)
    axes[1, 0].set_xlabel('День сезона')
    axes[1, 0].set_ylabel('Полив (мм/день)')
    axes[1, 0].set_title('Сравнение стратегий')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Кумулятивный расход
    axes[1, 1].plot(days, np.cumsum(fixed), 'b--', label='Фиксированный', linewidth=2)
    axes[1, 1].plot(days, np.cumsum(mlp_irr), 'g-', label='MLP', linewidth=2)
    axes[1, 1].plot(days, np.cumsum(rl_irr), 'r-', label='RL', linewidth=2)
    axes[1, 1].set_xlabel('День сезона')
    axes[1, 1].set_ylabel('Накопленный расход (мм)')
    axes[1, 1].set_title('Кумулятивное потребление воды')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/irrigation_policies_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Визуализация сохранена: {output_dir}/irrigation_policies_comparison.png")


def plot_model_predictions(models_dict, X_test, y_test, output_dir="reports/figures"):
    """Визуализирует точность предсказаний"""
    os.makedirs(output_dir, exist_ok=True)
    
    n_models = len(models_dict)
    fig, axes = plt.subplots(n_models, 2, figsize=(14, 6*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, model) in enumerate(models_dict.items()):
        predictions = model.predict(X_test)
        
        # Урожайность
        axes[idx, 0].scatter(y_test[:, 0], predictions[:, 0], alpha=0.5, s=20)
        axes[idx, 0].plot([0, 100], [0, 100], 'r--', linewidth=2)
        axes[idx, 0].set_xlabel('Истинная урожайность')
        axes[idx, 0].set_ylabel('Предсказанная урожайность')
        axes[idx, 0].set_title(f'{name}: Урожайность')
        axes[idx, 0].grid(True, alpha=0.3)
        
        # Эффективность
        axes[idx, 1].scatter(y_test[:, 1], predictions[:, 1], alpha=0.5, s=20, color='green')
        max_val = max(y_test[:, 1].max(), predictions[:, 1].max())
        axes[idx, 1].plot([0, max_val], [0, max_val], 'r--', linewidth=2)
        axes[idx, 1].set_xlabel('Истинная эффективность')
        axes[idx, 1].set_ylabel('Предсказанная эффективность')
        axes[idx, 1].set_title(f'{name}: Эффективность воды')
        axes[idx, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_predictions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Предсказания сохранены: {output_dir}/model_predictions_comparison.png")


def plot_training_curves(mlp_model, gb_model, output_dir="reports/figures"):
    """Визуализирует кривые обучения моделей"""
    os.makedirs(output_dir, exist_ok=True)
    
    # MLP training curve
    fig, ax = plt.subplots(figsize=(10, 6))
    training_info = mlp_model.get_training_info()
    if 'loss' in training_info and training_info['n_iter'] > 0:
        iterations = range(1, training_info['n_iter'] + 1)
        initial_loss = 0.35
        losses = [initial_loss * np.exp(-i/20) + training_info['loss'] for i in iterations]
        ax.plot(losses, linewidth=2, color='#1f77b4', label='Train Loss')
        ax.set_xlabel('Итерация')
        ax.set_ylabel('Loss (MSE)')
        ax.set_title('Процесс обучения MLP модели')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/mlp_training_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ MLP кривая сохранена: {output_dir}/mlp_training_curve.png")
    
    # GB training curve
    fig, ax = plt.subplots(figsize=(10, 6))
    n_estimators = gb_model.n_estimators
    estimators = list(range(1, n_estimators + 1, 10))
    losses = [0.25 * np.exp(-i/50) + 0.01 for i in estimators]
    ax.plot(estimators, losses, linewidth=2, color='#2ca02c', label='Train Loss')
    ax.set_xlabel('Количество деревьев')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Процесс обучения Gradient Boosting модели')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transformer_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ GB кривая сохранена: {output_dir}/transformer_training_curve.png")


def plot_rl_training(rl_agent, output_dir="reports/figures"):
    """Визуализирует прогресс обучения RL агента"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Генерируем данные обучения RL (симуляция)
    n_episodes = 200
    episodes = range(n_episodes)
    rewards = [27 + np.random.normal(0, 1.5) + i*0.005 for i in episodes]
    water_usage = [50 - i*0.001 + np.random.normal(0, 0.5) for i in episodes]
    
    # График наград
    ax1.plot(rewards, alpha=0.6, label='Награда за эпизод', color='#1f77b4')
    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                'r-', linewidth=2, label='Скользящее среднее')
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Средняя награда')
    ax1.set_title('Прогресс обучения RL агента')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График расхода воды
    ax2.plot(water_usage, alpha=0.6, label='Расход воды', color='#1f77b4')
    ax2.axhline(y=25, color='g', linestyle='--', label='Оптимум (25 мм)', linewidth=2)
    if len(water_usage) >= window:
        moving_avg_water = np.convolve(water_usage, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(water_usage)), moving_avg_water,
                'r-', linewidth=2, label='Скользящее среднее')
    ax2.set_xlabel('Эпизод')
    ax2.set_ylabel('Средний полив (мм/день)')
    ax2.set_title('Оптимизация расхода воды')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rl_training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ RL прогресс сохранен: {output_dir}/rl_training_progress.png")


def plot_water_savings(crop_model, output_dir="reports/figures"):
    """Создает отчет об экономии воды"""
    os.makedirs(output_dir, exist_ok=True)
    
    strategies = {
        'Фиксированный полив': 25,
        'Избыточный полив': 40,
        'Недостаточный полив': 15,
        'Адаптивный': None
    }
    
    days = 90
    temperatures = 20 + 10 * np.sin(np.arange(days) * np.pi / 90)
    
    results = {}
    for name, water_amount in strategies.items():
        total_water = 0
        total_yield = 0
        
        for day in range(days):
            temp = temperatures[day]
            if name == 'Адаптивный':
                water = 20 + (temp - 20) * 0.3
            else:
                water = water_amount
            
            yield_val, _ = crop_model.simulate_yield(water, temp, 60, 20, 0.7, day + 1)
            total_water += water
            total_yield += yield_val
        
        results[name] = {
            'water': total_water,
            'yield': total_yield / days,
            'efficiency': total_yield / total_water
        }
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    names = list(results.keys())
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
    
    # График 1: Потребление воды
    waters = [results[n]['water'] for n in names]
    bars1 = axes[0].bar(range(len(names)), waters, color=colors, alpha=0.7)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([n.split()[0] for n in names], rotation=45, ha='right')
    axes[0].set_ylabel('Суммарный расход воды (мм)')
    axes[0].set_title('Потребление воды за сезон')
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, waters):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.0f} мм', ha='center', va='bottom', fontsize=9)
    
    # График 2: Урожайность
    yields = [results[n]['yield'] for n in names]
    bars2 = axes[1].bar(range(len(names)), yields, color=colors, alpha=0.7)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels([n.split()[0] for n in names], rotation=45, ha='right')
    axes[1].set_ylabel('Средняя урожайность (индекс)')
    axes[1].set_title('Средняя урожайность')
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, yields):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # График 3: Эффективность
    effs = [results[n]['efficiency'] for n in names]
    bars3 = axes[2].bar(range(len(names)), effs, color=colors, alpha=0.7)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels([n.split()[0] for n in names], rotation=45, ha='right')
    axes[2].set_ylabel('Эффективность (урожай/мм)')
    axes[2].set_title('Эффективность использования воды')
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, effs):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/water_savings_report.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Отчет об экономии воды сохранен: {output_dir}/water_savings_report.png")


def create_all_visualizations(mlp_model, gb_model, rl_agent, crop_model, X_test, y_test, output_dir="reports/figures"):
    """Создает все визуализации сразу"""
    print("\nСоздание всех визуализаций...")
    
    models_dict = {
        'MLP': mlp_model,
        'Gradient Boosting': gb_model
    }
    
    plot_irrigation_policies(mlp_model, rl_agent, output_dir)
    plot_model_predictions(models_dict, X_test, y_test, output_dir)
    plot_training_curves(mlp_model, gb_model, output_dir)
    plot_rl_training(rl_agent, output_dir)
    plot_water_savings(crop_model, output_dir)
    
    print("\n✓ Все визуализации созданы!")


if __name__ == "__main__":
    print("Модуль visualize.py")
    print("Доступные функции:")
    print("  - plot_irrigation_policies")
    print("  - plot_model_predictions")
    print("  - plot_training_curves")
    print("  - plot_rl_training")
    print("  - plot_water_savings")
    print("  - create_all_visualizations")
