"""
Оценка качества моделей оптимизации полива
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os


def evaluate_models(models_dict, X_test, y_test, output_dir="reports/metrics"):
    """
    Оценивает все модели на тестовой выборке
    
    Args:
        models_dict: Словарь {название: модель}
        X_test: Тестовые признаки
        y_test: Тестовые метки
        output_dir: Директория для сохранения метрик
    """
    print("\n" + "="*60)
    print("ОЦЕНКА МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*60)
    
    results = {}
    
    for name, model in models_dict.items():
        print(f"\n{name}:")
        
        # Предсказания
        predictions = model.predict(X_test)
        
        # Метрики
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²:  {r2:.4f}")
        
        results[name] = {
            'MSE': float(mse),
            'MAE': float(mae),
            'R2': float(r2)
        }
    
    # Сохранение метрик
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/model_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Метрики сохранены в {output_dir}/model_metrics.json")
    
    return results


def compare_irrigation_strategies(crop_model, output_dir="reports/metrics"):
    """
    Сравнивает различные стратегии полива
    
    Args:
        crop_model: Модель симуляции растений
        output_dir: Директория для сохранения результатов
    """
    print("\n" + "="*60)
    print("СРАВНЕНИЕ СТРАТЕГИЙ ПОЛИВА")
    print("="*60)
    
    strategies = {
        'Фиксированный (25 мм)': lambda d, t: 25,
        'Избыточный (40 мм)': lambda d, t: 40,
        'Недостаточный (15 мм)': lambda d, t: 15,
        'Адаптивный': lambda d, t: 20 + (t - 20) * 0.3
    }
    
    days = 90
    temperatures = 20 + 10 * np.sin(np.arange(days) * np.pi / 90)
    
    results = {}
    
    for strategy_name, strategy_func in strategies.items():
        total_water = 0
        total_yield = 0
        
        for day in range(days):
            temp = temperatures[day]
            water = strategy_func(day, temp)
            
            yield_val, _ = crop_model.simulate_yield(
                water, temp, 60, 20, 0.7, day + 1
            )
            
            total_water += water
            total_yield += yield_val
        
        avg_yield = total_yield / days
        efficiency = total_yield / total_water
        
        results[strategy_name] = {
            'total_water': float(total_water),
            'avg_yield': float(avg_yield),
            'water_efficiency': float(efficiency)
        }
        
        print(f"\n{strategy_name}:")
        print(f"  Расход воды: {total_water:.1f} мм")
        print(f"  Средняя урожайность: {avg_yield:.2f}")
        print(f"  Эффективность: {efficiency:.4f}")
    
    # Сохранение
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/irrigation_strategies.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Результаты сохранены в {output_dir}/irrigation_strategies.json")
    
    return results


if __name__ == "__main__":
    print("Модуль evaluate.py")
    print("Используйте этот модуль через импорт")