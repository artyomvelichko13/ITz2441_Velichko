"""
Irrigation Optimization Package
Оптимизация графика полива с использованием ML и RL
"""

__version__ = "1.0.0"
__author__ = "Irrigation Optimization Team"
__email__ = "contact@irrigation-opt.com"

from .data_generator import CropWaterModel, SyntheticDataGenerator
from .models import IrrigationMLP, GradientBoostingModel, SimpleRLAgent
from .train import train_all_models
from .evaluate import evaluate_models, compare_irrigation_strategies
from .visualize import (plot_irrigation_policies, plot_model_predictions,
                        plot_training_curves, plot_rl_training, plot_water_savings,
                        create_all_visualizations)

__all__ = [
    'CropWaterModel',
    'SyntheticDataGenerator',
    'IrrigationMLP',
    'GradientBoostingModel',
    'SimpleRLAgent',
    'train_all_models',
    'evaluate_models',
    'compare_irrigation_strategies',
    'plot_irrigation_policies',
    'plot_model_predictions',
    'plot_training_curves',
    'plot_rl_training',
    'plot_water_savings',
    'create_all_visualizations',
]