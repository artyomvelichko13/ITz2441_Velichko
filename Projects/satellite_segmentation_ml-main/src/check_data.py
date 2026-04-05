"""
Проверка и информация о скачанных данных SpaceNet
Показывает, что программа найдёт и как будет использовать данные
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_data_structure():
    """
    Проверяет структуру данных и показывает, что найдено
    """
    
    print("="*80)
    print(" "*20 + "🔍 ПРОВЕРКА ДАННЫХ")
    print("="*80)
    
    data_dir = Path('data')
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    
    # Проверка 1: Существуют ли папки?
    print("\n📁 ПРОВЕРКА СТРУКТУРЫ:")
    print("-"*80)
    
    if not data_dir.exists():
        print("❌ Папка 'data/' НЕ НАЙДЕНА")
        print("\n💡 Решение:")
        print("   1. Скачайте данные: python download_spacenet_windows.py")
        print("   2. Или создайте синтетические: python create_local_data.py")
        return False
    else:
        print(f"✅ Папка 'data/' найдена: {data_dir.absolute()}")
    
    if not images_dir.exists():
        print(f"❌ Папка 'data/images/' НЕ НАЙДЕНА")
        return False
    else:
        print(f"✅ Папка 'data/images/' найдена")
    
    if not masks_dir.exists():
        print(f"❌ Папка 'data/masks/' НЕ НАЙДЕНА")
        return False
    else:
        print(f"✅ Папка 'data/masks/' найдена")
    
    # Проверка 2: Есть ли файлы?
    print("\n📊 КОЛИЧЕСТВО ФАЙЛОВ:")
    print("-"*80)
    
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.tif')) + list(images_dir.glob('*.tiff'))
    mask_files = list(masks_dir.glob('*.png')) + list(masks_dir.glob('*.tif')) + list(masks_dir.glob('*.tiff'))
    
    print(f"   Изображений: {len(image_files)}")
    print(f"   Масок: {len(mask_files)}")
    
    if len(image_files) == 0:
        print("\n❌ Изображения не найдены!")
        print("   Проверьте, что файлы скопированы в data/images/")
        return False
    
    if len(mask_files) == 0:
        print("\n❌ Маски не найдены!")
        print("   Проверьте, что файлы скопированы в data/masks/")
        return False
    
    # Проверка 3: Анализ первого файла
    print("\n🔬 АНАЛИЗ ДАННЫХ:")
    print("-"*80)
    
    sample_image = image_files[0]
    sample_mask = mask_files[0]
    
    print(f"   Пример изображения: {sample_image.name}")
    print(f"   Пример маски: {sample_mask.name}")
    
    # Загружаем пример
    try:
        img = cv2.imread(str(sample_image))
        if img is None:
            print(f"   ⚠️  Не удалось загрузить изображение")
            return False
        
        mask = cv2.imread(str(sample_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"   ⚠️  Не удалось загрузить маску")
            return False
        
        print(f"   Размер изображения: {img.shape}")
        print(f"   Размер маски: {mask.shape}")
        print(f"   Тип данных изображения: {img.dtype}")
        print(f"   Тип данных маски: {mask.dtype}")
        
        # Анализ классов в маске
        unique_classes = np.unique(mask)
        print(f"\n   Уникальные значения в маске: {unique_classes}")
        print(f"   Количество классов: {len(unique_classes)}")
        
        # Распределение классов
        print("\n   Распределение пикселей по классам:")
        for cls in unique_classes:
            count = np.sum(mask == cls)
            percentage = (count / mask.size) * 100
            print(f"      Класс {cls}: {count:,} пикселей ({percentage:.2f}%)")
        
    except Exception as e:
        print(f"   ❌ Ошибка при анализе: {e}")
        return False
    
    # Проверка 4: Совместимость форматов
    print("\n✅ СОВМЕСТИМОСТЬ:")
    print("-"*80)
    print("   ✅ Данные найдены и доступны для чтения")
    print("   ✅ Программа автоматически обнаружит эти данные")
    print("   ✅ При запуске train_segmentation.py будут использованы эти файлы")
    
    return True


def visualize_sample():
    """
    Показывает пример данных с визуализацией
    """
    
    print("\n" + "="*80)
    print(" "*20 + "🎨 ВИЗУАЛИЗАЦИЯ ПРИМЕРА")
    print("="*80)
    
    data_dir = Path('data')
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    
    image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.tif'))
    
    if not image_files:
        print("\n❌ Нет данных для визуализации")
        return
    
    # Берём первый пример
    sample_image = image_files[0]
    sample_mask = masks_dir / sample_image.name
    
    # Если маска с таким же именем не найдена, берём первую доступную
    if not sample_mask.exists():
        mask_files = list(masks_dir.glob('*.png')) + list(masks_dir.glob('*.tif'))
        if mask_files:
            sample_mask = mask_files[0]
        else:
            print("\n❌ Маски не найдены")
            return
    
    # Загружаем
    image = cv2.imread(str(sample_image))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(sample_mask), cv2.IMREAD_GRAYSCALE)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(image)
    axes[0].set_title(f'Изображение\n{sample_image.name}', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='tab10')
    axes[1].set_title(f'Маска сегментации\n{sample_mask.name}', fontsize=12)
    axes[1].axis('off')
    
    # Добавляем colorbar для маски
    cbar = plt.colorbar(axes[1].imshow(mask, cmap='tab10'), ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Класс', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    output_path = Path('data_visualization.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Визуализация сохранена: {output_path.absolute()}")
    print("\n   Откройте файл data_visualization.png чтобы посмотреть пример данных")


def show_will_be_used():
    """
    Показывает, какие данные будут использованы при запуске
    """
    
    print("\n" + "="*80)
    print(" "*15 + "📋 ЧТО БУДЕТ ИСПОЛЬЗОВАНО ПРИ ОБУЧЕНИИ")
    print("="*80)
    
    data_dir = Path('data')
    images_dir = data_dir / 'images'
    masks_dir = data_dir / 'masks'
    
    if (images_dir.exists() and masks_dir.exists()):
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.tif'))
        mask_files = list(masks_dir.glob('*.png')) + list(masks_dir.glob('*.tif'))
        
        if len(image_files) > 0 and len(mask_files) > 0:
            print("\n✅ При запуске train_segmentation.py или demo_pipeline.py:")
            print("="*80)
            print(f"\n   📂 Источник данных: РЕАЛЬНЫЕ ДАННЫЕ SPACENET")
            print(f"   📍 Расположение: {data_dir.absolute()}")
            print(f"   🖼️  Изображений: {len(image_files)}")
            print(f"   🎭 Масок: {len(mask_files)}")
            print(f"\n   Программа автоматически:")
            print(f"   ✅ Обнаружит папку data/")
            print(f"   ✅ Загрузит все изображения из data/images/")
            print(f"   ✅ Загрузит соответствующие маски из data/masks/")
            print(f"   ✅ Разделит на обучающую и валидационную выборки")
            print(f"   ✅ Начнёт обучение на реальных спутниковых снимках!")
            
            print("\n" + "="*80)
            print("🚀 ГОТОВО К ЗАПУСКУ!")
            print("="*80)
            print("\n   Запустите обучение:")
            print("   python train_segmentation.py")
            print("   или")
            print("   python demo_pipeline.py")
            
            return True
        else:
            print("\n⚠️  Папки найдены, но файлы отсутствуют")
            print("   Запустите: python download_spacenet_windows.py")
            return False
    else:
        print("\n❌ При запуске train_segmentation.py или demo_pipeline.py:")
        print("="*80)
        print(f"\n   📂 Источник данных: СИНТЕТИЧЕСКИЕ ДАННЫЕ (В ПАМЯТИ)")
        print(f"   ⚠️  Папка data/ не найдена")
        print(f"   ℹ️  Программа автоматически создаст временные данные")
        print(f"\n   Для использования реальных данных SpaceNet:")
        print(f"   1. Запустите: python download_spacenet_windows.py")
        print(f"   2. Подождите завершения скачивания (~15-30 минут)")
        print(f"   3. Запустите обучение снова")
        
        return False


def main():
    """
    Главная функция проверки
    """
    
    print("\n" + "="*80)
    print(" "*15 + "🔍 ПРОВЕРКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    print("="*80)
    print("\nЭтот скрипт проверит, найдёт ли программа ваши данные SpaceNet")
    
    # Проверка структуры
    if check_data_structure():
        # Показываем пример
        try:
            visualize_sample()
        except Exception as e:
            print(f"\n⚠️  Не удалось создать визуализацию: {e}")
        
        # Показываем, что будет использовано
        show_will_be_used()
        
        print("\n" + "="*80)
        print("✅ ВСЁ ГОТОВО!")
        print("="*80)
        print("\n   Данные найдены и готовы к использованию.")
        print("   Программа автоматически обнаружит и использует их.")
        
    else:
        print("\n" + "="*80)
        print("⚠️  ДАННЫЕ НЕ НАЙДЕНЫ")
        print("="*80)
        print("\n   💡 Варианты действий:")
        print("\n   1️⃣  Скачать реальные данные SpaceNet:")
        print("      python download_spacenet_windows.py")
        print("\n   2️⃣  Создать синтетические данные (быстро):")
        print("      python create_local_data.py")
        print("\n   3️⃣  Запустить с временными данными:")
        print("      python demo_pipeline.py")
        print("      (Данные будут созданы автоматически в памяти)")
        
        show_will_be_used()


if __name__ == '__main__':
    main()
