"""
Генератор синтетических данных для обучения моделей оптимизации полива
Основано на физиологических моделях роста растений
"""

import numpy as np
from typing import Tuple


class CropWaterModel:
    """
    Симулятор роста растений с учетом полива
    Моделирует физиологические процессы растений на основе научных исследований
    
    References:
        - Allen et al. (1998). FAO Irrigation and drainage paper 56
        - Steduto et al. (2012). FAO Irrigation and drainage paper 66
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Инициализация модели
        
        Args:
            random_seed: Seed для воспроизводимости
        """
        np.random.seed(random_seed)
        
        # Параметры культуры (томаты как пример)
        self.optimal_water = 25.0      # мм/день оптимальный полив
        self.min_water = 10.0          # минимум для выживания
        self.max_water = 50.0          # максимум без вреда
        
        # Коэффициенты влияния стресс-факторов
        self.water_stress_coef = 0.8
        self.overwater_coef = 0.6
        self.temp_stress_coef = 0.7
        
    def simulate_yield(
        self, 
        water: float, 
        temperature: float, 
        humidity: float, 
        solar_radiation: float,
        soil_moisture: float, 
        day_in_season: int
    ) -> Tuple[float, float]:
        """
        Симулирует урожайность и эффективность использования воды
        
        Args:
            water: Объем полива (мм/день)
            temperature: Температура воздуха (°C)
            humidity: Влажность воздуха (%)
            solar_radiation: Солнечная радиация (МДж/м²/день)
            soil_moisture: Влажность почвы (доля 0-1)
            day_in_season: День в сезоне (1-120)
            
        Returns:
            Tuple[float, float]: (индекс урожайности, эффективность воды)
        """
        
        # Базовая урожайность
        base_yield = 100.0
        
        # 1. Водный стресс (недостаток воды)
        if water < self.optimal_water:
            water_deficit = (self.optimal_water - water) / self.optimal_water
            water_stress = water_deficit * self.water_stress_coef
            base_yield *= (1 - water_stress)
        
        # 2. Стресс от избытка воды (переувлажнение)
        elif water > self.optimal_water:
            water_excess = (water - self.optimal_water) / self.optimal_water
            overwater_stress = min(water_excess * self.overwater_coef, 0.9)
            base_yield *= (1 - overwater_stress)
        
        # 3. Температурный стресс
        optimal_temp = 25.0
        if temperature < 15 or temperature > 35:
            temp_stress = abs(temperature - optimal_temp) / optimal_temp * self.temp_stress_coef
            base_yield *= (1 - min(temp_stress, 0.8))
        
        # 4. Влияние влажности воздуха
        if humidity < 40:
            humidity_stress = (40 - humidity) / 40 * 0.3
            base_yield *= (1 - humidity_stress)
        
        # 5. Влияние солнечной радиации
        optimal_radiation = 20.0  # МДж/м²/день
        radiation_factor = min(solar_radiation / optimal_radiation, 1.5) * 0.2 + 0.8
        base_yield *= radiation_factor
        
        # 6. Влияние влажности почвы
        optimal_soil_moisture = 0.7
        soil_factor = 1 - abs(soil_moisture - optimal_soil_moisture) * 0.5
        base_yield *= max(soil_factor, 0.5)
        
        # 7. Сезонный фактор (стадия роста)
        growth_stage_factor = self._get_growth_stage_factor(day_in_season)
        base_yield *= growth_stage_factor
        
        # Эффективность использования воды (урожай на единицу воды)
        water_efficiency = base_yield / max(water, 1.0)
        
        # Добавляем небольшой реалистичный шум
        noise = np.random.normal(0, 2)
        base_yield = max(0, min(100, base_yield + noise))
        
        return base_yield, water_efficiency
    
    def _get_growth_stage_factor(self, day: int) -> float:
        """
        Фактор стадии роста растения
        
        Args:
            day: День в сезоне
            
        Returns:
            float: Фактор продуктивности (0-1)
        """
        if day < 30:  # Начальная стадия (установление)
            return 0.6 + (day / 30) * 0.2
        elif day < 60:  # Вегетативная стадия
            return 0.8 + ((day - 30) / 30) * 0.15
        elif day < 90:  # Цветение и плодоношение (пик)
            return 0.95
        else:  # Созревание (снижение)
            return 0.95 - ((day - 90) / 30) * 0.15


class SyntheticDataGenerator:
    """
    Генератор синтетических данных для обучения моделей
    Создает реалистичные образцы на основе физиологической модели
    """
    
    def __init__(self, n_samples: int = 5000, random_seed: int = 42):
        """
        Инициализация генератора
        
        Args:
            n_samples: Количество образцов для генерации
            random_seed: Seed для воспроизводимости
        """
        self.n_samples = n_samples
        self.crop_model = CropWaterModel(random_seed)
        np.random.seed(random_seed)
        
    def generate_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Генерирует датасет с различными условиями полива
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                X: входные признаки [water, temp, humidity, radiation, soil_moisture, day]
                y: целевые значения [yield, water_efficiency]
        """
        
        X = []
        y = []
        
        print(f"Генерация {self.n_samples} образцов...")
        
        for i in range(self.n_samples):
            # Генерируем случайные условия с реалистичными распределениями
            water = np.random.uniform(5, 60)  # мм/день
            temperature = np.random.normal(25, 7)  # °C (нормальное распределение)
            temperature = np.clip(temperature, 10, 40)  # ограничиваем диапазон
            humidity = np.random.uniform(30, 90)  # %
            solar_radiation = np.random.uniform(10, 30)  # МДж/м²/день
            soil_moisture = np.random.uniform(0.3, 0.9)  # доля
            day_in_season = np.random.randint(1, 121)  # день сезона (1-120)
            
            # Симулируем результат через физиологическую модель
            yield_val, water_eff = self.crop_model.simulate_yield(
                water, temperature, humidity, solar_radiation, 
                soil_moisture, day_in_season
            )
            
            X.append([water, temperature, humidity, solar_radiation, 
                     soil_moisture, day_in_season])
            y.append([yield_val, water_eff])
            
            # Прогресс
            if (i + 1) % 1000 == 0:
                print(f"  Сгенерировано {i + 1}/{self.n_samples} образцов")
        
        print("✓ Генерация завершена")
        
        return np.array(X), np.array(y)
    
    def split_dataset(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Разделяет датасет на train/val/test
        
        Args:
            X: Входные признаки
            y: Целевые значения
            train_ratio: Доля обучающей выборки
            val_ratio: Доля валидационной выборки
            
        Returns:
            Tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        n_samples = len(X)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        
        return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Пример использования
    generator = SyntheticDataGenerator(n_samples=1000, random_seed=42)
    X, y = generator.generate_dataset()
    
    print(f"\nСтатистика датасета:")
    print(f"Форма X: {X.shape}")
    print(f"Форма y: {y.shape}")
    print(f"\nПримеры данных:")
    print(f"X[0]: {X[0]}")
    print(f"y[0]: {y[0]}")
