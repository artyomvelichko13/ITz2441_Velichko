"""
Скрипт для уменьшения объёма данных
Оставит только N случайных изображений и их маски
"""

import random
import shutil
from pathlib import Path
from tqdm import tqdm


def reduce_dataset(target_size=500, backup=True):
    """
    Уменьшает датасет до нужного размера
    
    Args:
        target_size: целевое количество примеров (по умолчанию 500)
        backup: создать резервную копию перед удалением
    """
    
    print("="*80)
    print(" "*20 + "✂️ УМЕНЬШЕНИЕ ОБЪЁМА ДАННЫХ")
    print("="*80)
    
    images_dir = Path('data/images')
    masks_dir = Path('data/masks')
    
    if not images_dir.exists() or not masks_dir.exists():
        print("\n❌ Папки data/images или data/masks не найдены!")
        return False
    
    # Получаем списки файлов
    image_files = sorted(list(images_dir.glob('*.png')) + 
                        list(images_dir.glob('*.tif')) + 
                        list(images_dir.glob('*.tiff')))
    
    mask_files = sorted(list(masks_dir.glob('*.png')) + 
                       list(masks_dir.glob('*.tif')) + 
                       list(masks_dir.glob('*.tiff')))
    
    current_size = len(image_files)
    
    print(f"\n📊 ТЕКУЩЕЕ СОСТОЯНИЕ:")
    print(f"   Изображений: {len(image_files)}")
    print(f"   Масок: {len(mask_files)}")
    
    if current_size <= target_size:
        print(f"\n✅ У вас уже {current_size} примеров (≤ {target_size})")
        print("   Уменьшение не требуется!")
        return True
    
    print(f"\n🎯 ЦЕЛЕВОЙ РАЗМЕР: {target_size} примеров")
    print(f"   Будет удалено: {current_size - target_size} файлов")
    
    # Создаём пары изображение-маска
    pairs = []
    for img in image_files:
        # Ищем соответствующую маску
        mask = masks_dir / f"{img.stem}.png"
        if not mask.exists():
            mask = masks_dir / f"{img.stem}.tif"
        if not mask.exists():
            mask = masks_dir / f"{img.stem}.tiff"
        
        if mask.exists():
            pairs.append((img, mask))
    
    print(f"   Найдено пар: {len(pairs)}")
    
    if len(pairs) < target_size:
        print(f"\n⚠️  Недостаточно пар для выборки!")
        target_size = len(pairs)
        print(f"   Будет оставлено: {target_size} пар")
    
    # Запрос подтверждения
    print(f"\n⚠️  ВНИМАНИЕ!")
    print(f"   Будет УДАЛЕНО {current_size - target_size} изображений и масок")
    print(f"   Останется только {target_size} случайных примеров")
    
    if backup:
        print(f"\n💾 Резервная копия будет создана в data_backup/")
    
    response = input(f"\n❓ Продолжить? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y', 'да']:
        print("\n❌ Отменено.")
        return False
    
    # Создаём backup если нужно
    if backup:
        print("\n💾 Создание резервной копии...")
        backup_dir = Path('data_backup')
        
        if backup_dir.exists():
            print("   ⚠️  data_backup уже существует, пропускаем копирование")
        else:
            backup_dir.mkdir(exist_ok=True)
            (backup_dir / 'images').mkdir(exist_ok=True)
            (backup_dir / 'masks').mkdir(exist_ok=True)
            
            print("   Копирование изображений...")
            for img in tqdm(image_files[:100], desc='Backup images'):  # Копируем первые 100 для примера
                shutil.copy2(img, backup_dir / 'images' / img.name)
            
            print("   Копирование масок...")
            for mask in tqdm(mask_files[:100], desc='Backup masks'):
                shutil.copy2(mask, backup_dir / 'masks' / mask.name)
            
            print(f"   ✅ Backup создан: {backup_dir.absolute()}")
    
    # Случайная выборка
    print(f"\n🎲 Случайная выборка {target_size} примеров...")
    selected_pairs = random.sample(pairs, target_size)
    selected_images = set(img for img, _ in selected_pairs)
    selected_masks = set(mask for _, mask in selected_pairs)
    
    # Удаляем НЕвыбранные файлы
    print(f"\n🗑️  Удаление лишних файлов...")
    
    deleted_images = 0
    for img in tqdm(image_files, desc='Очистка images'):
        if img not in selected_images:
            img.unlink()
            deleted_images += 1
    
    deleted_masks = 0
    for mask in tqdm(mask_files, desc='Очистка masks'):
        if mask not in selected_masks:
            mask.unlink()
            deleted_masks += 1
    
    # Финальная проверка
    final_images = len(list(images_dir.glob('*')))
    final_masks = len(list(masks_dir.glob('*')))
    
    print(f"\n✅ ГОТОВО!")
    print(f"   Удалено изображений: {deleted_images}")
    print(f"   Удалено масок: {deleted_masks}")
    print(f"   Осталось изображений: {final_images}")
    print(f"   Осталось масок: {final_masks}")
    
    return True


def show_time_estimate(num_samples):
    """
    Показывает примерное время обучения
    """
    
    print("\n" + "="*80)
    print(" "*20 + "⏱️ ОЦЕНКА ВРЕМЕНИ ОБУЧЕНИЯ")
    print("="*80)
    
    # Примерные оценки (на CPU)
    time_per_sample = 2  # секунд на образец на CPU
    
    epochs = 3
    time_per_epoch = (num_samples / 8) * time_per_sample  # batch_size = 8
    total_time = time_per_epoch * epochs
    
    print(f"\n📊 Для {num_samples} примеров:")
    print(f"   Время на 1 эпоху: ~{time_per_epoch/60:.1f} минут")
    print(f"   Время на {epochs} эпохи: ~{total_time/60:.1f} минут")
    
    print(f"\n💡 Рекомендации:")
    
    if num_samples > 2000:
        print(f"   ⚠️  {num_samples} примеров - ОЧЕНЬ МНОГО для CPU!")
        print(f"   Рекомендуется: 200-500 примеров")
    elif num_samples > 1000:
        print(f"   ⚠️  {num_samples} примеров - много для CPU")
        print(f"   Рекомендуется: 200-500 примеров")
    elif num_samples > 500:
        print(f"   ✅ {num_samples} примеров - приемлемо")
        print(f"   Для ещё более быстрого обучения: 200-300 примеров")
    else:
        print(f"   ✅ {num_samples} примеров - оптимально для CPU!")


def main():
    """
    Главная функция
    """
    
    print("\n" + "="*80)
    print(" "*15 + "✂️ УМЕНЬШЕНИЕ ДАТАСЕТА ДЛЯ БЫСТРОГО ОБУЧЕНИЯ")
    print("="*80)
    
    images_dir = Path('data/images')
    
    if not images_dir.exists():
        print("\n❌ Папка data/images не найдена!")
        return
    
    current_size = len(list(images_dir.glob('*.png')) + 
                      list(images_dir.glob('*.tif')) + 
                      list(images_dir.glob('*.tiff')))
    
    print(f"\n📊 Текущий размер датасета: {current_size} примеров")
    
    # Показываем оценку времени
    show_time_estimate(current_size)
    
    print("\n" + "="*80)
    print("ВЫБЕРИТЕ РАЗМЕР ДАТАСЕТА")
    print("="*80)
    
    print("\nРекомендуемые размеры:")
    print("   1. 100 примеров  - очень быстро (~3-4 минуты на эпоху)")
    print("   2. 200 примеров  - быстро (~6-8 минут на эпоху)")
    print("   3. 500 примеров  - умеренно (~15-20 минут на эпоху)")
    print("   4. 1000 примеров - медленно (~30-40 минут на эпоху)")
    print("   5. Свой вариант")
    print("   6. Отмена")
    
    choice = input("\nВаш выбор (1-6): ").strip()
    
    size_map = {
        '1': 100,
        '2': 200,
        '3': 500,
        '4': 1000,
    }
    
    if choice == '6':
        print("\n❌ Отменено.")
        return
    
    if choice == '5':
        try:
            target_size = int(input("Введите желаемое количество примеров: ").strip())
            if target_size <= 0:
                print("\n❌ Некорректное число!")
                return
        except ValueError:
            print("\n❌ Некорректный ввод!")
            return
    elif choice in size_map:
        target_size = size_map[choice]
    else:
        print("\n❌ Некорректный выбор!")
        return
    
    # Показываем новую оценку времени
    show_time_estimate(target_size)
    
    print("\n" + "="*80)
    
    # Уменьшаем датасет
    success = reduce_dataset(target_size=target_size, backup=True)
    
    if success:
        print("\n" + "="*80)
        print(" "*20 + "✅ ДАТАСЕТ УМЕНЬШЕН!")
        print("="*80)
        
        print(f"\n📊 НОВЫЙ РАЗМЕР: {target_size} примеров")
        print(f"💾 Резервная копия: data_backup/ (первые 100 примеров)")
        
        print("\n🚀 ТЕПЕРЬ МОЖНО ЗАПУСКАТЬ ОБУЧЕНИЕ:")
        print("   python train_segmentation.py")
        print("   или")
        print("   python demo_pipeline.py")
        
        print("\n⏱️  Ожидаемое время обучения (3 эпохи):")
        time_per_epoch = (target_size / 8) * 2 / 60
        total_time = time_per_epoch * 3
        print(f"   ~{total_time:.1f} минут (вместо {(current_size / 8) * 2 / 60 * 3:.1f} минут)")
        
        print("\n💡 Если нужно восстановить данные:")
        print("   Скопируйте файлы из data_backup/ обратно в data/")


if __name__ == '__main__':
    main()
