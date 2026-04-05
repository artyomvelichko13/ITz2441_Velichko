"""
Скрипт для загрузки и подготовки данных SpaceNet SN8 Floods
"""

import os
import tarfile
import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm
import json


def download_spacenet_data(output_dir='data_raw'):
    """
    Загрузка данных SpaceNet с AWS S3
    """
    print("="*60)
    print("ЗАГРУЗКА ДАННЫХ SPACENET SN8 FLOODS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        'Germany_Training': 's3://spacenet-dataset/spacenet/SN8_floods/tarballs/Germany_Training_Public.tar.gz',
        'Louisiana_East_Training': 's3://spacenet-dataset/spacenet/SN8_floods/tarballs/Louisiana-East_Training_Public.tar.gz',
        'Louisiana_West_Test': 's3://spacenet-dataset/spacenet/SN8_floods/tarballs/Louisiana-West_Test_Public.tar.gz'
    }
    
    for name, s3_path in datasets.items():
        output_file = os.path.join(output_dir, f"{name}.tar.gz")
        
        if os.path.exists(output_file):
            print(f"\n✓ {name} уже загружен")
            continue
        
        print(f"\nЗагрузка {name}...")
        print(f"Источник: {s3_path}")
        
        try:
            # Используем AWS CLI для загрузки
            cmd = ['aws', 's3', 'cp', s3_path, output_file, '--no-sign-request']
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Ждём завершения
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"✓ {name} успешно загружен")
            else:
                print(f"✗ Ошибка при загрузке {name}")
                print(f"Ошибка: {stderr}")
        
        except FileNotFoundError:
            print("\n⚠️  AWS CLI не установлен!")
            print("Установите AWS CLI:")
            print("  Ubuntu/Debian: sudo apt-get install awscli")
            print("  macOS: brew install awscli")
            print("  Windows: https://aws.amazon.com/cli/")
            print("\nИли скачайте данные вручную:")
            print(f"  wget {s3_path}")
            return False
    
    return True


def extract_archives(input_dir='data_raw', output_dir='data_extracted'):
    """
    Распаковка архивов
    """
    print("\n" + "="*60)
    print("РАСПАКОВКА АРХИВОВ")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    archives = list(Path(input_dir).glob('*.tar.gz'))
    
    if not archives:
        print("✗ Архивы не найдены в", input_dir)
        return False
    
    for archive_path in archives:
        extract_path = output_dir / archive_path.stem.replace('.tar', '')
        
        if extract_path.exists():
            print(f"\n✓ {archive_path.name} уже распакован")
            continue
        
        print(f"\nРаспаковка {archive_path.name}...")
        
        try:
            with tarfile.open(archive_path, 'r:gz') as tar:
                # Получаем список файлов
                members = tar.getmembers()
                
                # Распаковываем с прогресс-баром
                for member in tqdm(members, desc='Распаковка'):
                    tar.extract(member, extract_path)
            
            print(f"✓ {archive_path.name} успешно распакован")
        
        except Exception as e:
            print(f"✗ Ошибка при распаковке {archive_path.name}: {e}")
            return False
    
    return True


def organize_data(extracted_dir='data_extracted', output_dir='data'):
    """
    Организация данных в нужную структуру
    """
    print("\n" + "="*60)
    print("ОРГАНИЗАЦИЯ ДАННЫХ")
    print("="*60)
    
    images_dir = Path(output_dir) / 'images'
    masks_dir = Path(output_dir) / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_path = Path(extracted_dir)
    
    # Поиск всех изображений и масок
    image_files = []
    mask_files = []
    
    # SpaceNet SN8 обычно имеет структуру:
    # PRE-event/..., POST-event/..., annotations/...
    
    for root, dirs, files in os.walk(extracted_path):
        for file in files:
            file_path = Path(root) / file
            
            # Определяем тип файла
            if any(keyword in str(file_path).lower() for keyword in ['pre-event', 'post-event', 'image']):
                if file.endswith(('.png', '.tif', '.tiff', '.jpg')):
                    image_files.append(file_path)
            
            elif any(keyword in str(file_path).lower() for keyword in ['annotation', 'mask', 'label']):
                if file.endswith(('.png', '.tif', '.tiff')):
                    mask_files.append(file_path)
    
    print(f"\nНайдено изображений: {len(image_files)}")
    print(f"Найдено масок: {len(mask_files)}")
    
    # Копирование файлов
    if image_files:
        print("\nКопирование изображений...")
        for img_path in tqdm(image_files[:1000]):  # Ограничим для демо
            dest = images_dir / img_path.name
            if not dest.exists():
                shutil.copy2(img_path, dest)
    
    if mask_files:
        print("\nКопирование масок...")
        for mask_path in tqdm(mask_files[:1000]):  # Ограничим для демо
            dest = masks_dir / mask_path.name
            if not dest.exists():
                shutil.copy2(mask_path, dest)
    
    # Создаём метаданные
    metadata = {
        'num_images': len(list(images_dir.glob('*'))),
        'num_masks': len(list(masks_dir.glob('*'))),
        'image_dir': str(images_dir),
        'mask_dir': str(masks_dir)
    }
    
    with open(Path(output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Данные организованы в {output_dir}")
    print(f"  Изображений: {metadata['num_images']}")
    print(f"  Масок: {metadata['num_masks']}")
    
    return True


def create_sample_data(output_dir='data', num_samples=100, img_size=512):
    """
    Создание примерных синтетических данных (если реальные недоступны)
    """
    import numpy as np
    import cv2
    
    print("\n" + "="*60)
    print("СОЗДАНИЕ СИНТЕТИЧЕСКИХ ДАННЫХ")
    print("="*60)
    
    images_dir = Path(output_dir) / 'images'
    masks_dir = Path(output_dir) / 'masks'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nСоздание {num_samples} примеров...")
    
    for i in tqdm(range(num_samples)):
        # Генерация синтетического спутникового снимка
        image = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        
        # Добавляем текстуру
        noise = np.random.normal(0, 20, (img_size, img_size, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Генерация маски
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # Класс 1 - здания (красные квадраты)
        for _ in range(10):
            x, y = np.random.randint(0, img_size-40, 2)
            size = np.random.randint(20, 50)
            mask[y:y+size, x:x+size] = 1
            image[y:y+size, x:x+size] = [150, 150, 150]  # Серый цвет
        
        # Класс 2 - дороги (серые линии)
        for _ in range(5):
            if np.random.rand() > 0.5:
                y = np.random.randint(0, img_size)
                mask[y:y+8, :] = 2
                image[y:y+8, :] = [100, 100, 100]
            else:
                x = np.random.randint(0, img_size)
                mask[:, x:x+8] = 2
                image[:, x:x+8] = [100, 100, 100]
        
        # Класс 3 - вода (синие области)
        for _ in range(3):
            x, y = np.random.randint(30, img_size-30, 2)
            radius = np.random.randint(20, 60)
            # Создаём временные маски
            temp_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.circle(temp_mask, (x, y), radius, 1, -1)
            mask[temp_mask == 1] = 3
            # Рисуем на изображении
            cv2.circle(image, (x, y), radius, (50, 100, 200), -1)
        
        # Класс 4 - поля (зелёные области)
        field_mask = (mask == 0)
        mask[field_mask] = np.random.choice([0, 4], size=field_mask.sum(), p=[0.2, 0.8])
        image[mask == 4] = np.random.randint(50, 150, (np.sum(mask == 4), 3))
        image[mask == 4, 1] += 50  # Больше зелёного
        
        # Сохранение
        cv2.imwrite(str(images_dir / f'synthetic_{i:04d}.png'), image)
        cv2.imwrite(str(masks_dir / f'synthetic_{i:04d}.png'), mask)
    
    print(f"\n✓ Создано {num_samples} синтетических примеров")
    print(f"  Расположение: {output_dir}")


def main():
    """
    Главная функция для подготовки данных
    """
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
    print("="*60)
    
    # Выбор режима
    print("\nВыберите режим:")
    print("1. Загрузить реальные данные SpaceNet (требуется AWS CLI)")
    print("2. Создать синтетические данные для демонстрации")
    
    try:
        choice = input("\nВаш выбор (1 или 2): ").strip()
    except:
        choice = '2'  # По умолчанию синтетические данные
    
    if choice == '1':
        # Загрузка реальных данных
        if download_spacenet_data():
            if extract_archives():
                organize_data()
                print("\n" + "="*60)
                print("✓ ДАННЫЕ ГОТОВЫ К ИСПОЛЬЗОВАНИЮ")
                print("="*60)
            else:
                print("\n⚠️  Не удалось распаковать архивы")
                print("Создаём синтетические данные...")
                create_sample_data()
        else:
            print("\n⚠️  Не удалось загрузить данные")
            print("Создаём синтетические данные...")
            create_sample_data()
    else:
        # Создание синтетических данных
        create_sample_data(num_samples=500, img_size=512)
        print("\n" + "="*60)
        print("✓ СИНТЕТИЧЕСКИЕ ДАННЫЕ СОЗДАНЫ")
        print("="*60)
        print("\nДанные готовы к использованию!")
        print("Запустите: python train_segmentation.py")


if __name__ == '__main__':
    main()
