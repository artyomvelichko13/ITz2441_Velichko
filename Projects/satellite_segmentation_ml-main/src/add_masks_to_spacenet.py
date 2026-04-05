"""
Создание масок для СУЩЕСТВУЮЩИХ изображений SpaceNet
Не удаляет скачанные изображения, только добавляет маски к ним!
"""

import os
import warnings

# Подавление предупреждений OpenCV о TIFF тегах
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore')

import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def create_masks_for_existing_images():
    """
    Создаёт синтетические маски для уже скачанных изображений SpaceNet
    НЕ удаляет существующие изображения!
    """
    
    print("="*80)
    print(" "*15 + "🎨 СОЗДАНИЕ МАСОК ДЛЯ SPACENET ИЗОБРАЖЕНИЙ")
    print("="*80)
    
    images_dir = Path('data/images')
    masks_dir = Path('data/masks')
    
    # Проверка наличия изображений
    if not images_dir.exists():
        print("\n❌ Папка data/images не найдена!")
        print("   Сначала скачайте данные: python download_spacenet_windows.py")
        return False
    
    # Получаем список изображений
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff'))
    
    if len(image_files) == 0:
        print("\n❌ Изображения не найдены в data/images!")
        return False
    
    print(f"\n✅ Найдено изображений: {len(image_files)}")
    print(f"📂 Расположение: {images_dir.absolute()}")
    
    # Создаём папку для масок
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверяем, есть ли уже маски
    existing_masks = list(masks_dir.glob('*'))
    if len(existing_masks) > 0:
        print(f"\n⚠️  В data/masks уже есть {len(existing_masks)} файлов!")
        response = input("   Удалить существующие маски и создать новые? (yes/no): ").strip().lower()
        
        if response not in ['yes', 'y', 'да']:
            print("\n❌ Отменено пользователем.")
            return False
        
        # Удаляем старые маски
        for mask_file in existing_masks:
            mask_file.unlink()
        print(f"   ✅ Удалено {len(existing_masks)} старых масок")
    
    print(f"\n🎨 Создание масок для {len(image_files)} изображений...")
    print("   (Маски будут иметь те же имена, что и изображения)")
    
    # Создаём маски для каждого изображения
    for img_path in tqdm(image_files, desc='Генерация масок'):
        # Загружаем изображение для определения размера
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n⚠️  Не удалось загрузить: {img_path.name}")
            continue
        
        height, width = img.shape[:2]
        
        # Создаём синтетическую маску
        mask = generate_synthetic_mask(width, height)
        
        # Сохраняем маску с тем же именем
        mask_path = masks_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)
    
    # Финальная проверка
    final_masks = list(masks_dir.glob('*'))
    
    print(f"\n✅ ГОТОВО!")
    print(f"   Создано масок: {len(final_masks)}")
    print(f"   Расположение: {masks_dir.absolute()}")
    
    # Проверка соответствия
    if len(final_masks) == len(image_files):
        print(f"\n✅ ИДЕАЛЬНО! Количество масок совпадает с количеством изображений!")
    else:
        print(f"\n⚠️  Количество масок ({len(final_masks)}) != количество изображений ({len(image_files)})")
    
    return True


def generate_synthetic_mask(width, height):
    """
    Генерирует синтетическую маску заданного размера
    """
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Класс 1 - здания (случайные квадраты)
    num_buildings = np.random.randint(15, 30)
    for _ in range(num_buildings):
        if width > 100 and height > 100:
            x = np.random.randint(0, max(1, width - 50))
            y = np.random.randint(0, max(1, height - 50))
            size = np.random.randint(20, min(50, width//10, height//10))
            mask[y:min(y+size, height), x:min(x+size, width)] = 1
    
    # Класс 2 - дороги (линии)
    num_roads = np.random.randint(5, 10)
    for _ in range(num_roads):
        if np.random.rand() > 0.5:
            # Горизонтальная дорога
            y = np.random.randint(0, height)
            road_width = np.random.randint(5, 15)
            mask[y:min(y+road_width, height), :] = 2
        else:
            # Вертикальная дорога
            x = np.random.randint(0, width)
            road_width = np.random.randint(5, 15)
            mask[:, x:min(x+road_width, width)] = 2
    
    # Класс 3 - вода (круги/эллипсы)
    num_water = np.random.randint(3, 8)
    for _ in range(num_water):
        if width > 100 and height > 100:
            x = np.random.randint(50, width - 50)
            y = np.random.randint(50, height - 50)
            radius = np.random.randint(20, min(80, width//8, height//8))
            
            # Создаём временную маску
            temp_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.circle(temp_mask, (x, y), radius, 1, -1)
            mask[temp_mask == 1] = 3
    
    # Класс 4 - поля (заполняем оставшееся пространство)
    field_mask = (mask == 0)
    if field_mask.any():
        mask[field_mask] = np.random.choice([0, 4], size=field_mask.sum(), p=[0.3, 0.7])
    
    return mask


def visualize_example():
    """
    Показывает пример изображения с созданной маской
    """
    
    print("\n" + "="*80)
    print(" "*20 + "🖼️ ВИЗУАЛИЗАЦИЯ ПРИМЕРА")
    print("="*80)
    
    import matplotlib.pyplot as plt
    
    images_dir = Path('data/images')
    masks_dir = Path('data/masks')
    
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.tif'))
    
    if not image_files:
        print("\n❌ Изображения не найдены")
        return
    
    # Берём случайное изображение
    img_path = np.random.choice(image_files)
    mask_path = masks_dir / f"{img_path.stem}.png"
    
    if not mask_path.exists():
        print("\n❌ Маска не найдена")
        return
    
    # Загружаем
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    # Цвета для классов
    colors = np.array([
        [0, 0, 0],       # 0 - фон
        [255, 0, 0],     # 1 - здания
        [128, 128, 128], # 2 - дороги
        [0, 0, 255],     # 3 - вода
        [0, 255, 0]      # 4 - поля
    ])
    
    mask_colored = colors[mask]
    
    # Создаём overlay
    overlay = (image * 0.6 + mask_colored * 0.4).astype(np.uint8)
    
    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title(f'Изображение SpaceNet\n{img_path.name}', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(mask_colored.astype(np.uint8))
    axes[1].set_title('Созданная маска', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')
    
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
    
    output_path = Path('spacenet_with_mask_example.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Пример сохранён: {output_path.absolute()}")


def main():
    """
    Главная функция
    """
    
    print("\n" + "="*80)
    print(" "*10 + "🎨 ДОБАВЛЕНИЕ МАСОК К ИЗОБРАЖЕНИЯМ SPACENET")
    print("="*80)
    
    print("\n📋 ЧТО БУДЕТ СДЕЛАНО:")
    print("   1. Проверит наличие изображений в data/images")
    print("   2. Создаст синтетические маски для КАЖДОГО изображения")
    print("   3. Сохранит маски в data/masks")
    print("   4. Визуализирует пример")
    
    print("\n⚠️  ВАЖНО:")
    print("   • Существующие изображения НЕ УДАЛЯЮТСЯ")
    print("   • Создаются только маски")
    print("   • Имена масок совпадают с именами изображений")
    
    images_dir = Path('data/images')
    if images_dir.exists():
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.tif'))
        print(f"\n✅ Найдено изображений: {len(image_files)}")
        print(f"   Будет создано масок: {len(image_files)}")
    else:
        print(f"\n❌ Папка data/images не найдена!")
        print("   Сначала скачайте данные: python download_spacenet_windows.py")
        return
    
    response = input("\n❓ Продолжить? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y', 'да']:
        print("\n❌ Отменено.")
        return
    
    # Создаём маски
    success = create_masks_for_existing_images()
    
    if success:
        # Визуализация
        try:
            visualize_example()
        except Exception as e:
            print(f"\n⚠️  Не удалось создать визуализацию: {e}")
        
        print("\n" + "="*80)
        print(" "*20 + "✅ ВСЁ ГОТОВО!")
        print("="*80)
        
        print("\n📊 ИТОГОВАЯ СТРУКТУРА:")
        images_count = len(list(Path('data/images').glob('*')))
        masks_count = len(list(Path('data/masks').glob('*')))
        
        print(f"""
   data/
   ├── images/  ({images_count} файлов) ← ВАШИ SpaceNet изображения ✅
   └── masks/   ({masks_count} файлов)  ← НОВЫЕ синтетические маски ✅
        """)
        
        print("\n🚀 ТЕПЕРЬ МОЖНО ЗАПУСКАТЬ ОБУЧЕНИЕ:")
        print("   python train_segmentation.py")
        print("   или")
        print("   python demo_pipeline.py")
        
        print("\n💡 Программа автоматически обнаружит и использует:")
        print(f"   • {images_count} изображений SpaceNet")
        print(f"   • {masks_count} созданных масок")


if __name__ == '__main__':
    main()
