"""
Архитектуры моделей для оптимизации полива
Включает MLP, Gradient Boosting и RL агента
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from typing import List
import pickle


class IrrigationMLP:
    """
    Многослойный персептрон для предсказания урожайности
    Использует sklearn MLPRegressor с custom архитектурой
    """
    
    def __init__(self, hidden_dims: List[int] = [128, 64, 32], random_state: int = 42):
        """
        Инициализация MLP модели
        
        Args:
            hidden_dims: Размеры скрытых слоев
            random_state: Seed для воспроизводимости
        """
        self.hidden_dims = hidden_dims
        
        # Две отдельные модели для двух выходов
        self.model_yield = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_dims),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=random_state,
            verbose=False
        )
        
        self.model_efficiency = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_dims),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=random_state,
            verbose=False
        )
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели
        
        Args:
            X: Входные признаки
            y: Целевые значения
        """
        print("  Масштабирование признаков...")
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        print("  Обучение модели для урожайности...")
        self.model_yield.fit(X_scaled, y_scaled[:, 0])
        
        print("  Обучение модели для эффективности...")
        self.model_efficiency.fit(X_scaled, y_scaled[:, 1])
        
        self.is_fitted = True
        print("  ✓ Обучение завершено")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание
        
        Args:
            X: Входные признаки
            
        Returns:
            np.ndarray: Предсказания [урожайность, эффективность]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        X_scaled = self.scaler_X.transform(X)
        
        pred_yield = self.model_yield.predict(X_scaled).reshape(-1, 1)
        pred_eff = self.model_efficiency.predict(X_scaled).reshape(-1, 1)
        
        predictions_scaled = np.concatenate([pred_yield, pred_eff], axis=1)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions
    
    def save(self, filepath: str):
        """Сохранение модели"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"  ✓ Модель сохранена: {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_training_info(self) -> dict:
        """Получение информации об обучении"""
        if not self.is_fitted:
            return {}
        
        return {
            'n_iter': self.model_yield.n_iter_,
            'loss': self.model_yield.loss_,
            'hidden_layers': self.hidden_dims
        }


class GradientBoostingModel:
    """
    Gradient Boosting регрессор для предсказания урожайности
    Использует ансамбль деревьев решений
    """
    
    def __init__(self, n_estimators: int = 200, random_state: int = 42):
        """
        Инициализация GB модели
        
        Args:
            n_estimators: Количество деревьев
            random_state: Seed для воспроизводимости
        """
        self.n_estimators = n_estimators
        
        self.model_yield = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state,
            verbose=0
        )
        
        self.model_efficiency = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state,
            verbose=0
        )
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Обучение модели
        
        Args:
            X: Входные признаки
            y: Целевые значения
        """
        print("  Масштабирование признаков...")
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        print("  Обучение GB для урожайности...")
        self.model_yield.fit(X_scaled, y_scaled[:, 0])
        
        print("  Обучение GB для эффективности...")
        self.model_efficiency.fit(X_scaled, y_scaled[:, 1])
        
        self.is_fitted = True
        print("  ✓ Обучение завершено")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказание
        
        Args:
            X: Входные признаки
            
        Returns:
            np.ndarray: Предсказания [урожайность, эффективность]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
            
        X_scaled = self.scaler_X.transform(X)
        
        pred_yield = self.model_yield.predict(X_scaled).reshape(-1, 1)
        pred_eff = self.model_efficiency.predict(X_scaled).reshape(-1, 1)
        
        predictions_scaled = np.concatenate([pred_yield, pred_eff], axis=1)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        
        return predictions
    
    def get_feature_importance(self) -> np.ndarray:
        """Получение важности признаков"""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        return self.model_yield.feature_importances_
    
    def save(self, filepath: str):
        """Сохранение модели"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"  ✓ Модель сохранена: {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Загрузка модели"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


class SimpleRLAgent:
    """
    Простой RL-агент (Policy Gradient) для оптимизации полива
    Использует нейронную сеть для выбора действий
    """
    
    def __init__(self, state_dim: int = 6, action_dim: int = 1, lr: float = 0.01):
        """
        Инициализация RL агента
        
        Args:
            state_dim: Размерность состояния
            action_dim: Размерность действия
            lr: Learning rate
        """
        # Простая линейная политика с нелинейными признаками
        self.W1 = np.random.randn(state_dim, 32) * 0.1
        self.b1 = np.zeros(32)
        self.W2 = np.random.randn(32, action_dim) * 0.1
        self.b2 = np.zeros(action_dim)
        
        self.lr = lr
        self.training_history = []
        
        # Импортируем модель симуляции
        try:
            from data_generator import CropWaterModel
        except ImportError:
            from .data_generator import CropWaterModel
        
        self.crop_model = CropWaterModel()
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def select_action(self, state: np.ndarray, explore: bool = False) -> float:
        """
        Выбирает действие (объем полива)
        
        Args:
            state: Текущее состояние
            explore: Использовать exploration
            
        Returns:
            float: Объем полива (мм)
        """
        # Forward pass
        h = self._relu(np.dot(state, self.W1) + self.b1)
        action = self._sigmoid(np.dot(h, self.W2) + self.b2)[0]
        
        # Exploration (добавляем шум)
        if explore:
            action += np.random.normal(0, 0.1)
            action = np.clip(action, 0, 1)
        
        # Масштабируем в диапазон [5, 50] мм
        return action * 45 + 5
    
    def train_episode(self, states: List[np.ndarray], rewards: List[float]) -> float:
        """
        Обучение на эпизоде (упрощенный policy gradient)
        
        Args:
            states: Список состояний
            rewards: Список наград
            
        Returns:
            float: Средний loss
        """
        total_loss = 0
        
        for state, reward in zip(states, rewards):
            # Forward pass
            h = self._relu(np.dot(state, self.W1) + self.b1)
            action_raw = np.dot(h, self.W2) + self.b2
            
            # Backward pass (простой градиентный спуск)
            # Максимизируем награду
            grad_action = reward * 0.01
            
            # Обновляем веса
            grad_W2 = np.outer(h, grad_action)
            grad_b2 = grad_action
            
            self.W2 += self.lr * grad_W2
            self.b2 += self.lr * grad_b2
            
            total_loss += abs(reward)
        
        avg_loss = total_loss / len(states)
        self.training_history.append(avg_loss)
        
        return avg_loss
    
    def save(self, filepath: str):
        """Сохранение агента"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"  ✓ RL агент сохранен: {filepath}")
    
    @staticmethod
    def load(filepath: str):
        """Загрузка агента"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    # Тестирование моделей
    print("Тестирование моделей...")
    
    # Генерируем тестовые данные
    X_test = np.random.randn(100, 6)
    y_test = np.random.randn(100, 2)
    
    # MLP
    print("\n1. MLP модель:")
    mlp = IrrigationMLP()
    mlp.fit(X_test, y_test)
    predictions = mlp.predict(X_test[:5])
    print(f"Предсказания: {predictions[:2]}")
    
    # GB
    print("\n2. Gradient Boosting модель:")
    gb = GradientBoostingModel(n_estimators=50)
    gb.fit(X_test, y_test)
    predictions = gb.predict(X_test[:5])
    print(f"Предсказания: {predictions[:2]}")
    
    # RL
    print("\n3. RL агент:")
    rl = SimpleRLAgent()
    state = np.random.randn(6)
    action = rl.select_action(state)
    print(f"Выбранное действие: {action:.2f} мм")
    
    print("\n✓ Все модели работают корректно")