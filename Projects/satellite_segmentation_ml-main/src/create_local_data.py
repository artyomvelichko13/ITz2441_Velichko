"""
Скрипт для создания локальных данных в папке data/
Запустите этот скрипт ПЕРЕД обучением
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def create_local_data(num_samples=500, img_size=512, output_dir='data'):
    """
    Создаёт локальные синтетические данные для обучения
    
    Args:
        num_samples: количество примеров для создания
        img_size: размер изображений
        output_dir: папка для сохранения данных
    """
    
    print("="*70)
    print("СОЗДАНИЕ ЛОКАЛЬНЫХ ДАННЫХ")
    print("="*70)
    
    # Создаём директории
    images_dir = Path(output_dir) / 'images'
    masks_dir = Path(output_dir) / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Папки созданы:")
    print(f"   • Изображения: {images_dir.absolute()}")
    print(f"   • Маски: {masks_dir.absolute()}")
    
    print(f"\n🎨 Создание {num_samples} примеров...")
    print("   Классы: 0-Фон, 1-Здания, 2-Дороги, 3-Вода, 4-Поля")
    
    for i in tqdm(range(num_samples), desc='Генерация данных'):
        # Генерация синтетического спутникового снимка
        image = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        
        # Добавляем текстуру
        noise = np.random.normal(0, 20, (img_size, img_size, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Генерация маски
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Класс 1 - здания (красные квадраты)
        num_buildings = np.random.randint(8, 15)
        for _ in range(num_buildings):
            x, y = np.random.randint(0, img_size-40, 2)
            size = np.random.randint(20, 50)
            mask[y:y+size, x:x+size] = 1
            # Добавляем текстуру зданий
            image[y:y+size, x:x+size] = [150, 150, 150]  # Серый цвет
        
        # Класс 2 - дороги (серые линии)
        num_roads = np.random.randint(3, 7)
        for _ in range(num_roads):
            if np.random.rand() > 0.5:
                # Горизонтальная дорога
                y = np.random.randint(0, img_size)
                width = np.random.randint(8, 15)
                mask[y:y+width, :] = 2
                image[y:y+width, :] = [100, 100, 100]
            else:
                # Вертикальная дорога
                x = np.random.randint(0, img_size)
                width = np.random.randint(8, 15)
                mask[:, x:x+width] = 2
                image[:, x:x+width] = [100, 100, 100]
        
        # Класс 3 - вода (синие области)
        num_water = np.random.randint(2, 5)
        for _ in range(num_water):
            x, y = np.random.randint(30, img_size-30, 2)
            radius = np.random.randint(20, 60)
            # Создаём временную маску для круга
            temp_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.circle(temp_mask, (x, y), radius, 1, -1)
            mask[temp_mask == 1] = 3
            # Рисуем на изображении
            cv2.circle(image, (x, y), radius, (50, 100, 200), -1)
        
        # Класс 4 - поля (зелёные области)
        field_mask = (mask == 0)
        mask[field_mask] = np.random.choice([0, 4], size=field_mask.sum(), p=[0.2, 0.8])
        # Окрашиваем поля в зелёный
        field_pixels = mask == 4
        image[field_pixels] = np.random.randint(50, 150, (np.sum(field_pixels), 3))
        image[field_pixels, 1] += 50  # Больше зелёного канала
        
        # Сохранение
        image_path = images_dir / f'synthetic_{i:04d}.png'
        mask_path = masks_dir / f'synthetic_{i:04d}.png'
        
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), mask)
    
    print(f"\n✅ Данные созданы успешно!")
    print(f"   📊 Всего примеров: {num_samples}")
    print(f"   📏 Размер изображений: {img_size}x{img_size}")
    print(f"   💾 Общий размер: ~{(num_samples * img_size * img_size * 4 / 1024 / 1024):.1f} МБ")
    
    # Проверка
    image_files = list(images_dir.glob('*.png'))
    mask_files = list(masks_dir.glob('*.png'))
    
    print(f"\n🔍 Проверка:")
    print(f"   • Изображений: {len(image_files)}")
    print(f"   • Масок: {len(mask_files)}")
    
    if len(image_files) == num_samples and len(mask_files) == num_samples:
        print(f"\n✅ ВСЁ ГОТОВО!")
        print(f"\n🚀 Теперь можете запустить обучение:")
        print(f"   python train_segmentation.py")
        print(f"   python demo_pipeline.py")
    else:
        print(f"\n⚠️  Предупреждение: количество файлов не совпадает!")
    
    # Создаём README в папке data
    readme_path = Path(output_dir) / 'README.txt'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("ДАННЫЕ ДЛЯ ОБУЧЕНИЯ МОДЕЛИ СЕГМЕНТАЦИИ\n")
        f.write("="*50 + "\n\n")
        f.write(f"Количество примеров: {num_samples}\n")
        f.write(f"Размер изображений: {img_size}x{img_size}\n\n")
        f.write("СТРУКТУРА:\n")
        f.write("  data/\n")
        f.write("    ├── images/          # Исходные изображения\n")
        f.write("    │   └── synthetic_*.png\n")
        f.write("    ├── masks/           # Маски сегментации\n")
        f.write("    │   └── synthetic_*.png\n")
        f.write("    └── README.txt\n\n")
        f.write("КЛАССЫ СЕГМЕНТАЦИИ:\n")
        f.write("  0 - Фон (чёрный)\n")
        f.write("  1 - Здания (красный)\n")
        f.write("  2 - Дороги (серый)\n")
        f.write("  3 - Вода (синий)\n")
        f.write("  4 - Поля (зелёный)\n\n")
        f.write("Данные созданы автоматически скриптом create_local_data.py\n")
    
    print(f"\n📝 Создан файл README: {readme_path.absolute()}")
    
    return images_dir, masks_dir


def show_sample(data_dir='data'):
    """Показывает пример из созданных данных"""
    import matplotlib.pyplot as plt
    
    images_dir = Path(data_dir) / 'images'
    masks_dir = Path(data_dir) / 'masks'
    
    image_files = sorted(list(images_dir.glob('*.png')))
    
    if not image_files:
        print("❌ Данные не найдены! Сначала создайте их.")
        return
    
    # Загружаем первый пример
    image_path = image_files[0]
    mask_path = masks_dir / image_path.name
    
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Цвета для классов
    colors = np.array([
        [0, 0, 0],      # 0 - фон
        [255, 0, 0],    # 1 - здания
        [128, 128, 128], # 2 - дороги
        [0, 0, 255],    # 3 - вода
        [0, 255, 0]     # 4 - поля
    ])
    
    mask_colored = colors[mask]
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(image)
    axes[0].set_title('Изображение', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(mask_colored.astype(np.uint8))
    axes[1].set_title('Маска сегментации', fontsize=14)
    axes[1].axis('off')
    
    # Легенда
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
    plt.savefig('data/example_sample.png', dpi=150, bbox_inches='tight')
    print(f"\n🖼️  Пример сохранён: data/example_sample.png")
    plt.close()


def main():
    """Главная функция"""
    print("\n" + "="*70)
    print(" "*15 + "🎨 СОЗДАНИЕ ЛОКАЛЬНЫХ ДАННЫХ")
    print("="*70)
    
    print("\nЭтот скрипт создаст синтетические данные для обучения в папке 'data/'")
    print("\n📋 Параметры:")
    print("   • Количество примеров: 500")
    print("   • Размер изображений: 512x512")
    print("   • Классы: Фон, Здания, Дороги, Вода, Поля")
    
    input("\n⏎ Нажмите Enter для начала генерации... ")
    
    # Создаём данные
    create_local_data(num_samples=500, img_size=512, output_dir='data')
    
    # Показываем пример
    print("\n" + "="*70)
    print("СОЗДАНИЕ ПРИМЕРА")
    print("="*70)
    show_sample('data')
    
    print("\n" + "="*70)
    print("✅ ВСЁ ГОТОВО!")
    print("="*70)
    print("\n📂 Структура папок:")
    print("   lab1/")
    print("   ├── data/                    ← ЗДЕСЬ ВАШИ ДАННЫЕ")
    print("   │   ├── images/")
    print("   │   │   └── synthetic_*.png  (500 файлов)")
    print("   │   ├── masks/")
    print("   │   │   └── synthetic_*.png  (500 файлов)")
    print("   │   ├── README.txt")
    print("   │   └── example_sample.png   (пример)")
    print("   ├── train_segmentation.py")
    print("   ├── demo_pipeline.py")
    print("   └── ...")
    print("\n🚀 Следующий шаг:")
    print("   python demo_pipeline.py")
    print("   или")
    print("   python train_segmentation.py")


if __name__ == '__main__':
    main()
