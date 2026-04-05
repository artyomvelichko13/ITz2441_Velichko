"""
Скрипт обучения моделей оптимизации полива
Поддерживает MLP, Gradient Boosting и RL агента
"""

import argparse
import os
import sys
import numpy as np
from typing import Tuple
import time

from data_generator import SyntheticDataGenerator
from models import IrrigationMLP, GradientBoostingModel, SimpleRLAgent


def train_mlp(X_train, y_train, X_val, y_val, save_path: str) -> IrrigationMLP:
    """
    Обучает MLP модель
    
    Args:
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные
        save_path: Путь для сохранения модели
        
    Returns:
        IrrigationMLP: Обученная модель
    """
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ MLP МОДЕЛИ")
    print("="*60)
    
    start_time = time.time()
    
    model = IrrigationMLP(hidden_dims=[128, 64, 32])
    model.fit(X_train, y_train)
    
    # Валидация
    from sklearn.metrics import mean_squared_error, r2_score
    val_predictions = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    
    training_time = time.time() - start_time
    
    print(f"\nРезультаты:")
    print(f"  Время обучения: {training_time:.2f} сек")
    print(f"  Валидационный MSE: {val_mse:.4f}")
    print(f"  Валидационный R²: {val_r2:.4f}")
    
    # Сохранение
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    return model


def train_gradient_boosting(X_train, y_train, X_val, y_val, save_path: str) -> GradientBoostingModel:
    """
    Обучает Gradient Boosting модель
    
    Args:
        X_train, y_train: Обучающие данные
        X_val, y_val: Валидационные данные
        save_path: Путь для сохранения модели
        
    Returns:
        GradientBoostingModel: Обученная модель
    """
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ GRADIENT BOOSTING МОДЕЛИ")
    print("="*60)
    
    start_time = time.time()
    
    model = GradientBoostingModel(n_estimators=200)
    model.fit(X_train, y_train)
    
    # Валидация
    from sklearn.metrics import mean_squared_error, r2_score
    val_predictions = model.predict(X_val)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_r2 = r2_score(y_val, val_predictions)
    
    training_time = time.time() - start_time
    
    print(f"\nРезультаты:")
    print(f"  Время обучения: {training_time:.2f} сек")
    print(f"  Валидационный MSE: {val_mse:.4f}")
    print(f"  Валидационный R²: {val_r2:.4f}")
    
    # Feature importance
    feature_names = ['Полив', 'Температура', 'Влажность', 'Радиация', 'Почва', 'День']
    importance = model.get_feature_importance()
    print(f"\n  Важность признаков:")
    for name, imp in zip(feature_names, importance):
        print(f"    {name}: {imp:.4f}")
    
    # Сохранение
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    
    return model


def train_rl_agent(n_episodes: int, save_path: str) -> SimpleRLAgent:
    """
    Обучает RL агента
    
    Args:
        n_episodes: Количество эпизодов обучения
        save_path: Путь для сохранения агента
        
    Returns:
        SimpleRLAgent: Обученный агент
    """
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ RL АГЕНТА")
    print("="*60)
    
    start_time = time.time()
    
    agent = SimpleRLAgent()
    episode_rewards = []
    water_usage = []
    
    for episode in range(n_episodes):
        # Генерируем эпизод (сезон 90 дней)
        states = []
        rewards = []
        total_reward = 0
        total_water = 0
        
        for day in range(1, 90):
            # Случайные условия
            temperature = np.random.normal(25, 5)
            humidity = np.random.uniform(40, 80)
            solar_radiation = np.random.uniform(15, 25)
            soil_moisture = np.random.uniform(0.5, 0.8)
            
            state = np.array([0, temperature, humidity, solar_radiation, 
                            soil_moisture, day])
            
            # Агент выбирает действие
            water = agent.select_action(state, explore=(episode < n_episodes * 0.8))
            state[0] = water
            
            # Получаем награду
            yield_val, water_eff = agent.crop_model.simulate_yield(
                water, temperature, humidity, solar_radiation,
                soil_moisture, day
            )
            
            # Reward функция
            water_penalty = max(0, water - 30) * 0.5
            reward = yield_val + water_eff * 10 - water_penalty
            
            states.append(state)
            rewards.append(reward)
            total_reward += reward
            total_water += water
        
        # Обучение на эпизоде
        loss = agent.train_episode(states, rewards)
        
        episode_rewards.append(total_reward / len(states))
        water_usage.append(total_water / len(states))
        
        # Прогресс
        if (episode + 1) % 40 == 0:
            avg_reward = np.mean(episode_rewards[-40:])
            avg_water = np.mean(water_usage[-40:])
            print(f"  Эпизод {episode+1}/{n_episodes} - "
                  f"Награда: {avg_reward:.2f}, Полив: {avg_water:.2f} мм/день")
    
    training_time = time.time() - start_time
    
    print(f"\nРезультаты:")
    print(f"  Время обучения: {training_time:.2f} сек")
    print(f"  Финальная награда: {np.mean(episode_rewards[-20:]):.2f}")
    print(f"  Средний расход воды: {np.mean(water_usage[-20:]):.2f} мм/день")
    
    # Сохранение
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    
    return agent


def train_all_models(
    n_samples: int = 10000,
    n_rl_episodes: int = 200,
    output_dir: str = "models"
):
    """
    Обучает все модели
    
    Args:
        n_samples: Количество образцов для генерации
        n_rl_episodes: Количество эпизодов для RL
        output_dir: Директория для сохранения моделей
    """
    print("="*60)
    print("СИСТЕМА ОПТИМИЗАЦИИ ГРАФИКА ПОЛИВА")
    print("Обучение всех моделей")
    print("="*60)
    
    # 1. Генерация данных
    print("\n[1/4] Генерация синтетических данных...")
    generator = SyntheticDataGenerator(n_samples=n_samples, random_seed=42)
    X, y = generator.generate_dataset()
    
    X_train, y_train, X_val, y_val, X_test, y_test = generator.split_dataset(X, y)
    
    print(f"✓ Датасет готов:")
    print(f"  Train: {len(X_train)} образцов")
    print(f"  Val:   {len(X_val)} образцов")
    print(f"  Test:  {len(X_test)} образцов")
    
    # Сохранение данных
    data_dir = "data/processed"
    os.makedirs(data_dir, exist_ok=True)
    np.save(f"{data_dir}/X_train.npy", X_train)
    np.save(f"{data_dir}/y_train.npy", y_train)
    np.save(f"{data_dir}/X_val.npy", X_val)
    np.save(f"{data_dir}/y_val.npy", y_val)
    np.save(f"{data_dir}/X_test.npy", X_test)
    np.save(f"{data_dir}/y_test.npy", y_test)
    print(f"  ✓ Данные сохранены в {data_dir}/")
    
    # 2. Обучение MLP
    print("\n[2/4] Обучение MLP модели...")
    mlp_model = train_mlp(
        X_train, y_train, X_val, y_val,
        save_path=f"{output_dir}/mlp_model.pkl"
    )
    
    # 3. Обучение Gradient Boosting
    print("\n[3/4] Обучение Gradient Boosting модели...")
    gb_model = train_gradient_boosting(
        X_train, y_train, X_val, y_val,
        save_path=f"{output_dir}/gb_model.pkl"
    )
    
    # 4. Обучение RL агента
    print("\n[4/4] Обучение RL агента...")
    rl_agent = train_rl_agent(
        n_episodes=n_rl_episodes,
        save_path=f"{output_dir}/rl_agent.pkl"
    )
    
    print("\n" + "="*60)
    print("✓ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"\nМодели сохранены в директории: {output_dir}/")
    print(f"Данные сохранены в директории: data/processed/")
    
    return mlp_model, gb_model, rl_agent


def main():
    """Главная функция с поддержкой CLI"""
    parser = argparse.ArgumentParser(
        description="Обучение моделей оптимизации полива"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['mlp', 'gb', 'rl', 'all'],
        default='all',
        help='Модель для обучения (по умолчанию: all)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Количество образцов для генерации (по умолчанию: 10000)'
    )
    
    parser.add_argument(
        '--n-episodes',
        type=int,
        default=200,
        help='Количество эпизодов для RL (по умолчанию: 200)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Директория для сохранения моделей (по умолчанию: models)'
    )
    
    args = parser.parse_args()
    
    if args.model == 'all':
        train_all_models(
            n_samples=args.n_samples,
            n_rl_episodes=args.n_episodes,
            output_dir=args.output_dir
        )
    else:
        # Генерация данных
        print("Генерация данных...")
        generator = SyntheticDataGenerator(n_samples=args.n_samples)
        X, y = generator.generate_dataset()
        X_train, y_train, X_val, y_val, X_test, y_test = generator.split_dataset(X, y)
        
        # Обучение выбранной модели
        if args.model == 'mlp':
            train_mlp(X_train, y_train, X_val, y_val, 
                     f"{args.output_dir}/mlp_model.pkl")
        elif args.model == 'gb':
            train_gradient_boosting(X_train, y_train, X_val, y_val,
                                   f"{args.output_dir}/gb_model.pkl")
        elif args.model == 'rl':
            train_rl_agent(args.n_episodes, f"{args.output_dir}/rl_agent.pkl")


if __name__ == "__main__":
    main()
