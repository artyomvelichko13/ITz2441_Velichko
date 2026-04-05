"""
Скрипт для поиска и организации масок SpaceNet
SpaceNet имеет специфическую структуру - найдём маски правильно!
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm
import json


def find_spacenet_masks():
    """
    Ищет маски в распакованных данных SpaceNet
    """
    
    print("="*80)
    print(" "*20 + "🔍 ПОИСК МАСОК SPACENET")
    print("="*80)
    
    extracted_dir = Path('data_extracted')
    
    if not extracted_dir.exists():
        print("\n❌ Папка data_extracted не найдена!")
        print("   Сначала скачайте данные: python download_spacenet_windows.py")
        return
    
    print(f"\n📂 Поиск в: {extracted_dir.absolute()}")
    
    # Ищем все файлы во всех поддиректориях
    all_files = []
    
    print("\n🔍 Сканирование структуры данных...")
    for root, dirs, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith(('.png', '.tif', '.tiff', '.geojson', '.json')):
                file_path = Path(root) / file
                all_files.append(file_path)
    
    print(f"   Найдено файлов: {len(all_files)}")
    
    # Анализ структуры
    print("\n📊 АНАЛИЗ СТРУКТУРЫ:")
    print("-"*80)
    
    # Группируем по типам
    images = []
    masks = []
    labels = []
    other = []
    
    keywords_image = ['pre-event', 'post-event', 'image', 'img']
    keywords_mask = ['mask', 'label', 'building', 'road', 'flood', 'annotation']
    keywords_label = ['geojson', 'json']
    
    for file_path in all_files:
        path_str = str(file_path).lower()
        
        if any(keyword in path_str for keyword in keywords_label):
            labels.append(file_path)
        elif any(keyword in path_str for keyword in keywords_mask):
            # Дополнительная проверка - не является ли это изображением
            if not any(keyword in path_str for keyword in keywords_image):
                masks.append(file_path)
        elif any(keyword in path_str for keyword in keywords_image):
            images.append(file_path)
        else:
            other.append(file_path)
    
    print(f"   🖼️  Изображений: {len(images)}")
    print(f"   🎭 Масок (растр): {len(masks)}")
    print(f"   📝 Лейблов (векторных): {len(labels)}")
    print(f"   ❓ Других файлов: {len(other)}")
    
    # Показываем примеры путей
    if images:
        print(f"\n   Пример изображения:")
        print(f"      {images[0].relative_to(extracted_dir)}")
    
    if masks:
        print(f"\n   Пример маски:")
        print(f"      {masks[0].relative_to(extracted_dir)}")
    
    if labels:
        print(f"\n   Пример лейбла:")
        print(f"      {labels[0].relative_to(extracted_dir)}")
    
    # Показываем структуру директорий
    print("\n📁 СТРУКТУРА ДИРЕКТОРИЙ:")
    print("-"*80)
    
    unique_dirs = set()
    for f in all_files[:100]:  # Первые 100 для примера
        rel_path = f.relative_to(extracted_dir)
        if len(rel_path.parts) > 1:
            unique_dirs.add(rel_path.parts[0])
    
    for d in sorted(unique_dirs):
        subdir = extracted_dir / d
        if subdir.is_dir():
            subdirs = [x.name for x in subdir.iterdir() if x.is_dir()]
            print(f"   {d}/")
            for sub in subdirs[:5]:  # Первые 5
                print(f"      └── {sub}/")
    
    return images, masks, labels


def organize_spacenet_masks():
    """
    Организует найденные маски в правильную структуру
    """
    
    print("\n" + "="*80)
    print(" "*20 + "📦 ОРГАНИЗАЦИЯ МАСОК")
    print("="*80)
    
    images, masks, labels = find_spacenet_masks()
    
    if len(masks) == 0:
        print("\n⚠️  РАСТРОВЫЕ МАСКИ НЕ НАЙДЕНЫ!")
        print("\nВозможные причины:")
        print("   1. SpaceNet SN8 использует векторные аннотации (GeoJSON)")
        print("   2. Маски находятся в другой структуре директорий")
        print("   3. Маски имеют другие имена")
        
        if len(labels) > 0:
            print(f"\n💡 Найдены векторные аннотации: {len(labels)} файлов")
            print("   Это файлы .geojson с координатами объектов")
            print("   Их нужно конвертировать в растровые маски")
            
            response = input("\n❓ Показать пример содержимого GeoJSON? (yes/no): ").strip().lower()
            if response in ['yes', 'y', 'да']:
                show_geojson_example(labels[0])
        
        print("\n💡 РЕШЕНИЕ:")
        print("   Используйте синтетические данные для обучения:")
        print("   python create_local_data.py")
        
        return False
    
    # Копируем маски
    masks_dir = Path('data/masks')
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📋 Копирование {len(masks)} масок...")
    
    for i, mask_path in enumerate(tqdm(masks, desc='Маски')):
        dest = masks_dir / f"{mask_path.stem}_{i:04d}.png"
        try:
            shutil.copy2(mask_path, dest)
        except Exception as e:
            print(f"   ⚠️ Ошибка копирования {mask_path.name}: {e}")
    
    final_masks = len(list(masks_dir.glob('*')))
    
    print(f"\n✅ Скопировано масок: {final_masks}")
    
    return True


def show_geojson_example(geojson_path):
    """
    Показывает пример содержимого GeoJSON файла
    """
    import json
    
    print(f"\n📄 Файл: {geojson_path.name}")
    print("-"*80)
    
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        print(f"Тип: {data.get('type', 'unknown')}")
        
        if 'features' in data:
            print(f"Количество объектов: {len(data['features'])}")
            
            if len(data['features']) > 0:
                feature = data['features'][0]
                print(f"\nПример объекта:")
                print(f"   Тип геометрии: {feature.get('geometry', {}).get('type', 'unknown')}")
                print(f"   Свойства: {list(feature.get('properties', {}).keys())}")
        
        print("\n💡 Это векторная аннотация - нужна конвертация в растр")
        
    except Exception as e:
        print(f"Ошибка чтения: {e}")


def create_readme_with_findings():
    """
    Создаёт README с информацией о найденных данных
    """
    
    readme_path = Path('SPACENET_DATA_INFO.txt')
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("ИНФОРМАЦИЯ О ДАННЫХ SPACENET SN8\n")
        f.write("="*80 + "\n\n")
        f.write("СТАТУС: Изображения скачаны, но растровые маски НЕ НАЙДЕНЫ\n\n")
        f.write("ПРИЧИНА:\n")
        f.write("SpaceNet SN8 Floods использует векторные аннотации (GeoJSON),\n")
        f.write("а не готовые растровые маски для сегментации.\n\n")
        f.write("ЧТО СКАЧАНО:\n")
        f.write("- Спутниковые снимки (PRE-event и POST-event)\n")
        f.write("- Векторные аннотации наводнений (GeoJSON)\n")
        f.write("- Векторные аннотации зданий (GeoJSON)\n\n")
        f.write("ЧТО НУЖНО ДЛЯ ОБУЧЕНИЯ:\n")
        f.write("1. Конвертировать GeoJSON в растровые маски\n")
        f.write("2. Или использовать синтетические данные\n\n")
        f.write("РЕКОМЕНДАЦИЯ:\n")
        f.write("Используйте синтетические данные для обучения:\n")
        f.write("   python create_local_data.py\n\n")
        f.write("Синтетические данные специально созданы для этой задачи\n")
        f.write("и содержат готовые растровые маски для классов:\n")
        f.write("- Фон\n")
        f.write("- Здания\n")
        f.write("- Дороги\n")
        f.write("- Вода\n")
        f.write("- Поля\n")
    
    print(f"\n📝 Создан файл: {readme_path.absolute()}")


def main():
    """
    Главная функция
    """
    
    print("\n" + "="*80)
    print(" "*15 + "🔍 АНАЛИЗ ДАННЫХ SPACENET")
    print("="*80)
    
    print("\nЭтот скрипт проанализирует структуру скачанных данных SpaceNet")
    print("и попытается найти маски для обучения.")
    
    input("\n⏎ Нажмите Enter для начала анализа... ")
    
    # Анализ и организация
    success = organize_spacenet_masks()
    
    # Создаём информационный файл
    create_readme_with_findings()
    
    if not success:
        print("\n" + "="*80)
        print(" "*20 + "⚠️ МАСКИ НЕ НАЙДЕНЫ")
        print("="*80)
        
        print("\n💡 РЕКОМЕНДУЕМЫЕ ДЕЙСТВИЯ:")
        print("\n1️⃣  Используйте синтетические данные (РЕКОМЕНДУЕТСЯ):")
        print("      python create_local_data.py")
        print("      # Создаст 500 примеров с готовыми масками за 2-3 минуты")
        
        print("\n2️⃣  Или конвертируйте GeoJSON в растры (сложно):")
        print("      # Требует дополнительных библиотек: rasterio, shapely, geopandas")
        print("      # Требует знания географических координат")
        
        print("\n3️⃣  Или найдите другой датасет с готовыми масками:")
        print("      # EuroSAT")
        print("      # DeepGlobe")
        print("      # Inria Aerial Image Labeling")
        
        print("\n" + "="*80)
        print("Подробная информация сохранена в SPACENET_DATA_INFO.txt")
        print("="*80)
    else:
        print("\n" + "="*80)
        print(" "*20 + "✅ УСПЕШНО!")
        print("="*80)
        
        print("\nМаски найдены и организованы!")
        print("Теперь можно запускать обучение:")
        print("   python train_segmentation.py")


if __name__ == '__main__':
    main()
