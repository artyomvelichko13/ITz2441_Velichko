"""
Инференс обученной модели на новых спутниковых снимках
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from train_segmentation import UNet
import argparse


def load_model(model_path, num_classes=5, device='cpu'):
    """Загрузка обученной модели"""
    model = UNet(in_channels=3, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, image_path, device, img_size=256):
    """Предсказание для одного изображения"""
    # Загрузка изображения
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Предобработка
    image_resized = cv2.resize(image, (img_size, img_size))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Возвращаем предсказание к оригинальному размеру
    pred_resized = cv2.resize(pred.astype(np.uint8), 
                              (original_size[1], original_size[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    return image, pred_resized


def visualize_prediction(image, mask, save_path=None):
    """Визуализация предсказания"""
    # Цвета для классов
    colors = np.array([
        [0, 0, 0],      # 0 - фон (черный)
        [255, 0, 0],    # 1 - здания (красный)
        [128, 128, 128], # 2 - дороги (серый)
        [0, 0, 255],    # 3 - вода (синий)
        [0, 255, 0]     # 4 - поля (зеленый)
    ])
    
    # Создаём цветную маску
    mask_colored = colors[mask]
    
    # Создаём наложение (overlay)
    overlay = cv2.addWeighted(image, 0.6, mask_colored.astype(np.uint8), 0.4, 0)
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Оригинальное изображение', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(mask_colored.astype(np.uint8))
    axes[1].set_title('Сегментация', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Наложение', fontsize=14)
    axes[2].axis('off')
    
    # Добавляем легенду
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc=colors[0]/255, label='Фон'),
        plt.Rectangle((0,0),1,1, fc=colors[1]/255, label='Здания'),
        plt.Rectangle((0,0),1,1, fc=colors[2]/255, label='Дороги'),
        plt.Rectangle((0,0),1,1, fc=colors[3]/255, label='Вода'),
        plt.Rectangle((0,0),1,1, fc=colors[4]/255, label='Поля')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, 
               bbox_to_anchor=(0.5, 0.98), fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Результат сохранён в {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_statistics(mask, num_classes=5):
    """Вычисление статистики по классам"""
    total_pixels = mask.size
    
    class_names = ['Фон', 'Здания', 'Дороги', 'Вода', 'Поля']
    
    print("\n" + "="*50)
    print("СТАТИСТИКА СЕГМЕНТАЦИИ")
    print("="*50)
    
    for cls in range(num_classes):
        count = np.sum(mask == cls)
        percentage = (count / total_pixels) * 100
        print(f"{class_names[cls]}: {count:,} пикселей ({percentage:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Инференс модели сегментации')
    parser.add_argument('--image', type=str, required=True, 
                       help='Путь к изображению')
    parser.add_argument('--model', type=str, default='models/best_unet_model.pth',
                       help='Путь к модели')
    parser.add_argument('--output', type=str, default='outputs/inference_result.png',
                       help='Путь для сохранения результата')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Размер изображения для модели')
    
    args = parser.parse_args()
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загрузка модели
    print(f"Загрузка модели из {args.model}...")
    model = load_model(args.model, device=device)
    print("✓ Модель загружена")
    
    # Предсказание
    print(f"Обработка изображения {args.image}...")
    image, prediction = predict_image(model, args.image, device, args.img_size)
    print("✓ Предсказание выполнено")
    
    # Визуализация
    visualize_prediction(image, prediction, args.output)
    
    # Статистика
    calculate_statistics(prediction)


if __name__ == '__main__':
    main()
