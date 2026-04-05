"""
Пример полного пайплайна: от данных до результатов
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from train_segmentation import UNet, SyntheticDataset, calculate_iou, calculate_dice
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def demo_full_pipeline():
    """
    Демонстрация полного пайплайна работы с моделью
    """
    print("="*70)
    print(" "*15 + "🛰️  ДЕМОНСТРАЦИЯ ПОЛНОГО ПАЙПЛАЙНА")
    print("="*70)
    
    # Параметры
    NUM_CLASSES = 5
    IMG_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 3  # Небольшое количество для демо
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n📋 ПАРАМЕТРЫ:")
    print(f"   • Устройство: {DEVICE}")
    print(f"   • Размер изображения: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   • Batch size: {BATCH_SIZE}")
    print(f"   • Количество эпох: {NUM_EPOCHS}")
    print(f"   • Количество классов: {NUM_CLASSES}")
    
    # ==================== ШАГ 1: ДАННЫЕ ====================
    print("\n" + "="*70)
    print("ШАГ 1: СОЗДАНИЕ ДАТАСЕТА")
    print("="*70)
    
    train_dataset = SyntheticDataset(num_samples=200, img_size=IMG_SIZE)
    val_dataset = SyntheticDataset(num_samples=50, img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Обучающая выборка: {len(train_dataset)} примеров")
    print(f"✓ Валидационная выборка: {len(val_dataset)} примеров")
    
    # ==================== ШАГ 2: МОДЕЛЬ ====================
    print("\n" + "="*70)
    print("ШАГ 2: СОЗДАНИЕ МОДЕЛИ U-NET")
    print("="*70)
    
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Модель создана")
    print(f"✓ Всего параметров: {total_params:,}")
    
    # ==================== ШАГ 3: ОБУЧЕНИЕ ====================
    print("\n" + "="*70)
    print("ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ")
    print("="*70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    history = {'train_loss': [], 'val_loss': [], 'val_iou': []}
    
    for epoch in range(NUM_EPOCHS):
        # Обучение
        model.train()
        train_loss = 0.0
        
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        all_ious = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                ious = calculate_iou(preds, masks, NUM_CLASSES)
                all_ious.append(ious)
        
        val_loss /= len(val_loader)
        mean_iou = np.nanmean(all_ious)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(mean_iou)
        
        print(f"\nЭпоха {epoch+1}/{NUM_EPOCHS}:")
        print(f"   • Train Loss: {train_loss:.4f}")
        print(f"   • Val Loss: {val_loss:.4f}")
        print(f"   • Val IoU: {mean_iou:.4f}")
    
    print("\n✓ Обучение завершено!")
    
    # ==================== ШАГ 4: ОЦЕНКА ====================
    print("\n" + "="*70)
    print("ШАГ 4: ОЦЕНКА КАЧЕСТВА МОДЕЛИ")
    print("="*70)
    
    model.eval()
    all_ious = []
    all_dice = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            ious = calculate_iou(preds, masks, NUM_CLASSES)
            dice = calculate_dice(preds, masks, NUM_CLASSES)
            
            all_ious.append(ious)
            all_dice.append(dice)
    
    avg_ious = np.nanmean(all_ious, axis=0)
    avg_dice = np.nanmean(all_dice, axis=0)
    
    class_names = ['Фон', 'Здания', 'Дороги', 'Вода', 'Поля']
    
    print("\n📊 ФИНАЛЬНЫЕ МЕТРИКИ:")
    print("-"*70)
    print(f"{'Класс':<15} {'IoU':>10} {'Dice':>10}")
    print("-"*70)
    for name, iou, dice in zip(class_names, avg_ious, avg_dice):
        print(f"{name:<15} {iou:>10.4f} {dice:>10.4f}")
    print("-"*70)
    print(f"{'СРЕДНЕЕ':<15} {np.nanmean(avg_ious):>10.4f} {np.nanmean(avg_dice):>10.4f}")
    print("-"*70)
    
    # ==================== ШАГ 5: ВИЗУАЛИЗАЦИЯ ====================
    print("\n" + "="*70)
    print("ШАГ 5: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("="*70)
    
    # Создаём директорию для результатов
    import os
    os.makedirs('outputs/demo', exist_ok=True)
    
    # Визуализация обучения
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_title('Loss во время обучения')
    axes[0].set_xlabel('Эпоха')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['val_iou'], label='Val IoU', marker='o', color='green')
    axes[1].set_title('IoU во время обучения')
    axes[1].set_xlabel('Эпоха')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/demo/training_curves.png', dpi=150)
    print("✓ График обучения: outputs/demo/training_curves.png")
    plt.close()
    
    # Визуализация предсказаний
    colors = np.array([
        [0, 0, 0],      # Фон
        [255, 0, 0],    # Здания
        [128, 128, 128], # Дороги
        [0, 0, 255],    # Вода
        [0, 255, 0]     # Поля
    ])
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    model.eval()
    with torch.no_grad():
        for i in range(4):
            idx = np.random.randint(0, len(val_dataset))
            image, mask = val_dataset[idx]
            
            image_tensor = image.unsqueeze(0).to(DEVICE)
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            img_display = image.permute(1, 2, 0).numpy()
            mask_colored = colors[mask.numpy()]
            pred_colored = colors[pred]
            
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Изображение')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_colored.astype(np.uint8))
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_colored.astype(np.uint8))
            axes[i, 2].set_title('Предсказание')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/demo/predictions.png', dpi=150)
    print("✓ Примеры предсказаний: outputs/demo/predictions.png")
    plt.close()
    
    # Метрики по классам
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    x_pos = np.arange(len(class_names))
    
    axes[0].bar(x_pos, avg_ious, color=['black', 'red', 'gray', 'blue', 'green'])
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(class_names, rotation=45)
    axes[0].set_title('IoU по классам')
    axes[0].set_ylabel('IoU')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, axis='y')
    
    axes[1].bar(x_pos, avg_dice, color=['black', 'red', 'gray', 'blue', 'green'])
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].set_title('Dice Coefficient по классам')
    axes[1].set_ylabel('Dice')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/demo/metrics_by_class.png', dpi=150)
    print("✓ Метрики по классам: outputs/demo/metrics_by_class.png")
    plt.close()
    
    # ==================== ШАГ 6: СОХРАНЕНИЕ ====================
    print("\n" + "="*70)
    print("ШАГ 6: СОХРАНЕНИЕ МОДЕЛИ")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/demo_model.pth')
    print("✓ Модель сохранена: models/demo_model.pth")
    
    # ==================== ЗАВЕРШЕНИЕ ====================
    print("\n" + "="*70)
    print("✅ ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("="*70)
    print("\n📁 Созданные файлы:")
    print("   • models/demo_model.pth - обученная модель")
    print("   • outputs/demo/training_curves.png - графики обучения")
    print("   • outputs/demo/predictions.png - примеры предсказаний")
    print("   • outputs/demo/metrics_by_class.png - метрики по классам")
    print("\n🚀 Следующие шаги:")
    print("   1. Запустите train_segmentation.py для полного обучения")
    print("   2. Используйте inference.py для предсказаний на новых данных")
    print("   3. Изучите alternative_models.py для других архитектур")
    print("   4. Откройте segmentation_notebook.ipynb для интерактивной работы")
    print("="*70)


if __name__ == '__main__':
    demo_full_pipeline()
