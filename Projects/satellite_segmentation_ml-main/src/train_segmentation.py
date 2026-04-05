"""
Сегментация полей по спутниковым снимкам
Модель: U-Net
Датасет: SpaceNet SN8 Floods
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import cv2

# ==================== МОДЕЛЬ U-NET ====================

class DoubleConv(nn.Module):
    """Двойная свёртка: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    U-Net архитектура для семантической сегментации
    """
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder (Upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output layer
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.out(dec1)


# ==================== DATASET ====================

class SpaceNetDataset(Dataset):
    """
    Датасет для SpaceNet SN8 Floods
    Классы: 0-фон, 1-здания, 2-дороги, 3-вода, 4-поля
    """
    def __init__(self, image_dir, mask_dir, transform=None, img_size=256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.img_size = img_size
        
        # Получаем список файлов изображений
        self.images = sorted(list(self.image_dir.glob('*.png')) + 
                           list(self.image_dir.glob('*.tif')))
        
        print(f"Найдено изображений: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Загрузка изображения
        img_path = self.images[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Загрузка маски
        mask_path = self.mask_dir / img_path.name
        if not mask_path.exists():
            # Альтернативные имена файлов
            mask_path = self.mask_dir / img_path.name.replace('.png', '_mask.png')
        
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Создаём пустую маску если файл не найден
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Изменение размера
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), 
                         interpolation=cv2.INTER_NEAREST)
        
        # Преобразования
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        mask = torch.from_numpy(mask).long()
        
        return image, mask


class SyntheticDataset(Dataset):
    """
    Синтетический датасет для демонстрации (если реальные данные недоступны)
    """
    def __init__(self, num_samples=1000, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Генерация синтетического изображения
        image = np.random.rand(self.img_size, self.img_size, 3).astype(np.float32)
        
        # Генерация синтетической маски с различными классами
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        
        # Добавляем различные регионы
        # Класс 1 - здания (квадраты)
        for _ in range(5):
            x, y = np.random.randint(0, self.img_size-20, 2)
            size = np.random.randint(10, 30)
            mask[y:y+size, x:x+size] = 1
        
        # Класс 2 - дороги (линии)
        for _ in range(3):
            if np.random.rand() > 0.5:
                y = np.random.randint(0, self.img_size)
                mask[y:y+5, :] = 2
            else:
                x = np.random.randint(0, self.img_size)
                mask[:, x:x+5] = 2
        
        # Класс 3 - вода (круги)
        for _ in range(2):
            x, y = np.random.randint(20, self.img_size-20, 2)
            radius = np.random.randint(15, 40)
            # Создаём временную маску для круга
            temp_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.circle(temp_mask, (x, y), radius, 1, -1)
            mask[temp_mask == 1] = 3
        
        # Класс 4 - поля (большие регионы)
        mask[mask == 0] = np.random.choice([0, 4], size=(mask == 0).sum(), p=[0.3, 0.7])
        
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        
        return image, mask


# ==================== МЕТРИКИ ====================

def calculate_iou(pred, target, num_classes):
    """Вычисление IoU (Intersection over Union) для каждого класса"""
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection / union).item()
        ious.append(iou)
    
    return ious


def calculate_dice(pred, target, num_classes):
    """Вычисление Dice coefficient для каждого класса"""
    dice_scores = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection = (pred_cls & target_cls).sum().float()
        total = pred_cls.sum() + target_cls.sum()
        
        if total == 0:
            dice = float('nan')
        else:
            dice = (2.0 * intersection / total).item()
        dice_scores.append(dice)
    
    return dice_scores


# ==================== ОБУЧЕНИЕ ====================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Один эпох обучения"""
    model.train()
    running_loss = 0.0
    
    for images, masks in tqdm(dataloader, desc='Training'):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, num_classes):
    """Валидация модели"""
    model.eval()
    running_loss = 0.0
    all_ious = []
    all_dice = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()
            
            # Получаем предсказания
            preds = torch.argmax(outputs, dim=1)
            
            # Вычисляем метрики
            ious = calculate_iou(preds, masks, num_classes)
            dice = calculate_dice(preds, masks, num_classes)
            
            all_ious.append(ious)
            all_dice.append(dice)
    
    # Усредняем метрики
    avg_ious = np.nanmean(all_ious, axis=0)
    avg_dice = np.nanmean(all_dice, axis=0)
    avg_loss = running_loss / len(dataloader)
    
    return avg_loss, avg_ious, avg_dice


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def visualize_predictions(model, dataset, device, num_samples=4, save_path='outputs/predictions.png'):
    """Визуализация предсказаний модели"""
    model.eval()
    
    # Цвета для классов
    colors = np.array([
        [0, 0, 0],      # 0 - фон (черный)
        [255, 0, 0],    # 1 - здания (красный)
        [128, 128, 128], # 2 - дороги (серый)
        [0, 0, 255],    # 3 - вода (синий)
        [0, 255, 0]     # 4 - поля (зеленый)
    ])
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(0, len(dataset))
            image, mask = dataset[idx]
            
            # Предсказание
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            # Преобразуем изображение для отображения
            img_display = image.permute(1, 2, 0).cpu().numpy()
            
            # Создаём цветные маски
            mask_colored = colors[mask.numpy()]
            pred_colored = colors[pred]
            
            # Отображение
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Оригинальное изображение')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask_colored.astype(np.uint8))
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_colored.astype(np.uint8))
            axes[i, 2].set_title('Предсказание модели')
            axes[i, 2].axis('off')
    
    # Добавляем легенду
    legend_elements = [
        plt.Rectangle((0,0),1,1, fc=colors[0]/255, label='Фон'),
        plt.Rectangle((0,0),1,1, fc=colors[1]/255, label='Здания'),
        plt.Rectangle((0,0),1,1, fc=colors[2]/255, label='Дороги'),
        plt.Rectangle((0,0),1,1, fc=colors[3]/255, label='Вода'),
        plt.Rectangle((0,0),1,1, fc=colors[4]/255, label='Поля')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5, 
               bbox_to_anchor=(0.5, 1.02), fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Визуализация сохранена в {save_path}")
    plt.close()


def plot_training_history(history, save_path='outputs/training_history.png'):
    """Построение графиков обучения"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Mean IoU
    axes[0, 1].plot(history['mean_iou'], label='Mean IoU')
    axes[0, 1].set_title('Mean IoU')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Mean Dice
    axes[1, 0].plot(history['mean_dice'], label='Mean Dice', color='green')
    axes[1, 0].set_title('Mean Dice Coefficient')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # IoU по классам (последняя эпоха)
    class_names = ['Фон', 'Здания', 'Дороги', 'Вода', 'Поля']
    last_iou = history['class_iou'][-1]
    axes[1, 1].bar(class_names, last_iou)
    axes[1, 1].set_title('IoU по классам (последняя эпоха)')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"График обучения сохранён в {save_path}")
    plt.close()


# ==================== ГЛАВНАЯ ФУНКЦИЯ ====================

def main():
    # Параметры
    NUM_CLASSES = 5
    IMG_SIZE = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Создаём директории
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # ==================== ЗАГРУЗКА ДАННЫХ ====================
    print("\n" + "="*50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("="*50)
    
    # Проверяем наличие реальных данных
    data_dir = Path('data')
    
    if (data_dir / 'images').exists() and (data_dir / 'masks').exists():
        print("Используются реальные данные SpaceNet")
        train_dataset = SpaceNetDataset(
            image_dir=data_dir / 'images',
            mask_dir=data_dir / 'masks',
            img_size=IMG_SIZE
        )
        val_dataset = SpaceNetDataset(
            image_dir=data_dir / 'images',
            mask_dir=data_dir / 'masks',
            img_size=IMG_SIZE
        )
    else:
        print("⚠️  Реальные данные не найдены. Используются синтетические данные для демонстрации.")
        print("Для использования реальных данных:")
        print("1. Скачайте данные: aws s3 cp s3://spacenet-dataset/spacenet/SN8_floods/tarballs/Germany_Training_Public.tar.gz .")
        print("2. Распакуйте в data/images и data/masks")
        
        train_dataset = SyntheticDataset(num_samples=800, img_size=IMG_SIZE)
        val_dataset = SyntheticDataset(num_samples=200, img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"Размер обучающей выборки: {len(train_dataset)}")
    print(f"Размер валидационной выборки: {len(val_dataset)}")
    
    # ==================== СОЗДАНИЕ МОДЕЛИ ====================
    print("\n" + "="*50)
    print("СОЗДАНИЕ МОДЕЛИ U-NET")
    print("="*50)
    
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(device)
    
    # Подсчёт параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Всего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # ==================== ОБУЧЕНИЕ ====================
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ МОДЕЛИ")
    print("="*50)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'mean_iou': [],
        'mean_dice': [],
        'class_iou': [],
        'class_dice': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nЭпоха {epoch+1}/{NUM_EPOCHS}")
        print("-" * 50)
        
        # Обучение
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Валидация
        val_loss, val_ious, val_dice = validate(model, val_loader, criterion, device, NUM_CLASSES)
        
        # Обновление learning rate
        scheduler.step(val_loss)
        
        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mean_iou'].append(np.nanmean(val_ious))
        history['mean_dice'].append(np.nanmean(val_dice))
        history['class_iou'].append(val_ious)
        history['class_dice'].append(val_dice)
        
        # Вывод метрик
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Mean IoU: {np.nanmean(val_ious):.4f}")
        print(f"Mean Dice: {np.nanmean(val_dice):.4f}")
        print(f"IoU по классам: {[f'{iou:.3f}' for iou in val_ious]}")
        print(f"Dice по классам: {[f'{dice:.3f}' for dice in val_dice]}")
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_unet_model.pth')
            print(f"✓ Модель сохранена (Val Loss: {val_loss:.4f})")
    
    # ==================== ВИЗУАЛИЗАЦИЯ ====================
    print("\n" + "="*50)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("="*50)
    
    # Загрузка лучшей модели
    model.load_state_dict(torch.load('models/best_unet_model.pth'))
    
    # Визуализация предсказаний
    visualize_predictions(model, val_dataset, device, num_samples=4)
    
    # График обучения
    plot_training_history(history)
    
    # ==================== ФИНАЛЬНЫЕ МЕТРИКИ ====================
    print("\n" + "="*50)
    print("ФИНАЛЬНЫЕ МЕТРИКИ")
    print("="*50)
    
    class_names = ['Фон', 'Здания', 'Дороги', 'Вода', 'Поля']
    
    print(f"\nМетрики на валидационной выборке:")
    print(f"Mean IoU: {history['mean_iou'][-1]:.4f}")
    print(f"Mean Dice: {history['mean_dice'][-1]:.4f}")
    print("\nIoU по классам:")
    for name, iou in zip(class_names, history['class_iou'][-1]):
        print(f"  {name}: {iou:.4f}")
    print("\nDice по классам:")
    for name, dice in zip(class_names, history['class_dice'][-1]):
        print(f"  {name}: {dice:.4f}")
    
    # Сохранение истории
    with open('outputs/training_history.json', 'w', encoding='utf-8') as f:
        # Преобразуем numpy значения в обычные числа
        history_serializable = {
            'train_loss': [float(x) for x in history['train_loss']],
            'val_loss': [float(x) for x in history['val_loss']],
            'mean_iou': [float(x) for x in history['mean_iou']],
            'mean_dice': [float(x) for x in history['mean_dice']],
            'class_iou': [[float(x) for x in epoch] for epoch in history['class_iou']],
            'class_dice': [[float(x) for x in epoch] for epoch in history['class_dice']]
        }
        json.dump(history_serializable, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Обучение завершено!")
    print(f"✓ Модель сохранена в models/best_unet_model.pth")
    print(f"✓ Визуализация сохранена в outputs/")


if __name__ == '__main__':
    main()
