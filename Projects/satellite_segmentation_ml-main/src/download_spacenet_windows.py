"""
Скрипт для скачивания и подготовки данных SpaceNet на Windows 11
Автоматически скачает, распакует и организует данные для обучения
"""

import os
import sys
import subprocess
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import urllib.request


class DownloadProgressBar(tqdm):
    """Прогресс-бар для скачивания"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """
    Скачивает файл с прогресс-баром
    """
    print(f"\n📥 Скачивание: {output_path.name}")
    print(f"   Источник: {url}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    print(f"✅ Скачано: {output_path.name} ({output_path.stat().st_size / 1024 / 1024:.1f} МБ)")


def extract_tar_gz(tar_path, extract_to):
    """
    Распаковывает .tar.gz архив
    """
    print(f"\n📦 Распаковка: {tar_path.name}")
    
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            members = tar.getmembers()
            
            # Показываем прогресс
            for member in tqdm(members, desc='Извлечение файлов'):
                tar.extract(member, extract_to)
        
        print(f"✅ Распаковано в: {extract_to}")
        return True
    
    except Exception as e:
        print(f"❌ Ошибка при распаковке: {e}")
        return False


def organize_spacenet_data(extracted_dir, output_dir):
    """
    Организует данные SpaceNet в нужную структуру
    """
    print(f"\n📁 Организация данных...")
    
    images_dir = Path(output_dir) / 'images'
    masks_dir = Path(output_dir) / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_path = Path(extracted_dir)
    
    # Ищем все изображения и маски
    image_files = []
    mask_files = []
    
    print("   🔍 Поиск файлов...")
    
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            file_path = Path(root) / file
            
            # SpaceNet структура: PRE-event, POST-event, annotations
            if file.endswith(('.png', '.tif', '.tiff')):
                # Изображения
                if any(keyword in str(file_path).lower() for keyword in ['pre-event', 'post-event', 'images']):
                    if 'mask' not in file.lower() and 'label' not in file.lower():
                        image_files.append(file_path)
                
                # Маски
                elif any(keyword in str(file_path).lower() for keyword in ['mask', 'label', 'annotations', 'building']):
                    mask_files.append(file_path)
    
    print(f"   📊 Найдено изображений: {len(image_files)}")
    print(f"   📊 Найдено масок: {len(mask_files)}")
    
    # Копируем файлы
    if image_files:
        print("\n   📋 Копирование изображений...")
        for i, img_path in enumerate(tqdm(image_files[:1000], desc='Изображения')):
            dest = images_dir / f"{img_path.stem}_{i:04d}.png"
            try:
                shutil.copy2(img_path, dest)
            except Exception as e:
                print(f"   ⚠ Ошибка копирования {img_path.name}: {e}")
    
    if mask_files:
        print("\n   📋 Копирование масок...")
        for i, mask_path in enumerate(tqdm(mask_files[:1000], desc='Маски')):
            dest = masks_dir / f"{mask_path.stem}_{i:04d}.png"
            try:
                shutil.copy2(mask_path, dest)
            except Exception as e:
                print(f"   ⚠ Ошибка копирования {mask_path.name}: {e}")
    
    # Проверка
    final_images = len(list(images_dir.glob('*')))
    final_masks = len(list(masks_dir.glob('*')))
    
    print(f"\n   ✅ Скопировано изображений: {final_images}")
    print(f"   ✅ Скопировано масок: {final_masks}")
    
    return final_images, final_masks


def download_spacenet_windows():
    """
    Главная функция для скачивания данных SpaceNet на Windows
    """
    
    print("="*80)
    print(" "*20 + "🛰️ СКАЧИВАНИЕ ДАННЫХ SPACENET")
    print("="*80)
    
    print("\nЭтот скрипт скачает и подготовит реальные спутниковые снимки SpaceNet SN8.")
    print("\n⚠️  ВНИМАНИЕ:")
    print("   • Размер данных: ~3-5 ГБ")
    print("   • Время скачивания: 10-30 минут (зависит от скорости интернета)")
    print("   • Требуется свободное место: ~10 ГБ")
    
    print("\n📦 Будут скачаны:")
    print("   1. Germany Training Data (~1.5 ГБ)")
    print("   2. Louisiana-East Training Data (~1.5 ГБ)")
    print("   3. Louisiana-West Test Data (~1.0 ГБ)")
    
    response = input("\n❓ Продолжить скачивание? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y', 'да']:
        print("\n❌ Отменено пользователем.")
        print("\n💡 Альтернатива: используйте синтетические данные:")
        print("   python create_local_data.py")
        return
    
    # Создаём директории
    raw_dir = Path('data_raw')
    extracted_dir = Path('data_extracted')
    output_dir = Path('data')
    
    raw_dir.mkdir(exist_ok=True)
    extracted_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Данные для скачивания
    datasets = {
        'Germany_Training': 'https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Germany_Training_Public.tar.gz',
        'Louisiana_East_Training': 'https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Louisiana-East_Training_Public.tar.gz',
        'Louisiana_West_Test': 'https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Louisiana-West_Test_Public.tar.gz'
    }
    
    print("\n" + "="*80)
    print("ШАГ 1: СКАЧИВАНИЕ ДАННЫХ")
    print("="*80)
    
    # Скачиваем данные
    for name, url in datasets.items():
        archive_path = raw_dir / f"{name}.tar.gz"
        
        if archive_path.exists():
            print(f"\n✓ {name} уже скачан: {archive_path}")
            continue
        
        try:
            download_file(url, archive_path)
        except Exception as e:
            print(f"\n❌ Ошибка при скачивании {name}: {e}")
            print(f"   Попробуйте скачать вручную: {url}")
            continue
    
    print("\n" + "="*80)
    print("ШАГ 2: РАСПАКОВКА АРХИВОВ")
    print("="*80)
    
    # Распаковываем архивы
    for name in datasets.keys():
        archive_path = raw_dir / f"{name}.tar.gz"
        extract_path = extracted_dir / name
        
        if not archive_path.exists():
            print(f"\n⚠ Пропущен {name} (файл не найден)")
            continue
        
        if extract_path.exists() and any(extract_path.iterdir()):
            print(f"\n✓ {name} уже распакован: {extract_path}")
            continue
        
        extract_tar_gz(archive_path, extract_path)
    
    print("\n" + "="*80)
    print("ШАГ 3: ОРГАНИЗАЦИЯ ДАННЫХ")
    print("="*80)
    
    # Организуем все данные в одну структуру
    total_images = 0
    total_masks = 0
    
    for name in datasets.keys():
        extract_path = extracted_dir / name
        
        if not extract_path.exists():
            continue
        
        print(f"\n📂 Обработка: {name}")
        images, masks = organize_spacenet_data(extract_path, output_dir)
        total_images += images
        total_masks += masks
    
    print("\n" + "="*80)
    print("✅ ГОТОВО!")
    print("="*80)
    
    print(f"\n📊 ИТОГО:")
    print(f"   • Всего изображений: {total_images}")
    print(f"   • Всего масок: {total_masks}")
    print(f"   • Расположение: {output_dir.absolute()}")
    
    print(f"\n📁 Структура:")
    print(f"   {output_dir.absolute()}/")
    print(f"   ├── images/  ({total_images} файлов)")
    print(f"   └── masks/   ({total_masks} файлов)")
    
    # Создаём README
    readme_path = output_dir / 'README.txt'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("ДАННЫЕ SPACENET SN8 FLOODS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Источник: SpaceNet Challenge 8\n")
        f.write(f"Тип: Спутниковые снимки с разметкой наводнений\n")
        f.write(f"Всего изображений: {total_images}\n")
        f.write(f"Всего масок: {total_masks}\n\n")
        f.write("Датасеты:\n")
        for name in datasets.keys():
            f.write(f"  • {name}\n")
        f.write("\nСкачано и подготовлено автоматически\n")
    
    print(f"\n📝 Создан README: {readme_path.absolute()}")
    
    # Очистка (опционально)
    print("\n🧹 ОЧИСТКА")
    response = input("Удалить временные файлы (архивы и распакованные данные)? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y', 'да']:
        print("\n   Удаление временных файлов...")
        try:
            if raw_dir.exists():
                shutil.rmtree(raw_dir)
                print(f"   ✅ Удалено: {raw_dir}")
            if extracted_dir.exists():
                shutil.rmtree(extracted_dir)
                print(f"   ✅ Удалено: {extracted_dir}")
            print("\n   💾 Освобождено ~8-10 ГБ")
        except Exception as e:
            print(f"   ⚠ Ошибка при удалении: {e}")
    else:
        print("\n   Временные файлы сохранены в:")
        print(f"     • {raw_dir.absolute()}")
        print(f"     • {extracted_dir.absolute()}")
    
    print("\n" + "="*80)
    print("🚀 ТЕПЕРЬ МОЖНО ЗАПУСКАТЬ ОБУЧЕНИЕ!")
    print("="*80)
    print("\n   python train_segmentation.py")
    print("   или")
    print("   python demo_pipeline.py")


def check_space():
    """Проверяет доступное место на диске"""
    import shutil
    
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    
    print(f"\n💾 Свободное место на диске: {free_gb:.1f} ГБ")
    
    if free_gb < 15:
        print("⚠️  ПРЕДУПРЕЖДЕНИЕ: Мало свободного места!")
        print("   Рекомендуется минимум 15 ГБ для комфортной работы.")
        return False
    
    return True


if __name__ == '__main__':
    print("\n" + "="*80)
    print(" "*15 + "🛰️ СКАЧИВАНИЕ ДАННЫХ SPACENET ДЛЯ WINDOWS 11")
    print("="*80)
    
    # Проверка места
    if not check_space():
        response = input("\nПродолжить несмотря на предупреждение? (yes/no): ").strip().lower()
        if response not in ['yes', 'y', 'да']:
            print("\n❌ Отменено.")
            print("\n💡 Используйте синтетические данные: python create_local_data.py")
            sys.exit(0)
    
    # Скачивание
    try:
        download_spacenet_windows()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем (Ctrl+C)")
        print("   Вы можете запустить скрипт снова - уже скачанные файлы будут пропущены.")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        print("\n💡 Если возникли проблемы:")
        print("   1. Проверьте подключение к интернету")
        print("   2. Убедитесь, что есть свободное место на диске")
        print("   3. Или используйте синтетические данные: python create_local_data.py")
