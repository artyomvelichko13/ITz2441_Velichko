"""
Скрипт для отключения предупреждений OpenCV во всех файлах проекта
Запустите один раз, чтобы добавить подавление предупреждений
"""

import os
from pathlib import Path


def add_warning_suppression(file_path):
    """
    Добавляет код подавления предупреждений в начало Python файла
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Проверяем, не добавлено ли уже
    if 'OPENCV_LOG_LEVEL' in content:
        return False, "Уже исправлено"
    
    # Ищем первый import
    lines = content.split('\n')
    
    # Находим позицию после docstring и перед первым import
    insert_pos = 0
    in_docstring = False
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Пропускаем docstring
        if '"""' in stripped or "'''" in stripped:
            if not in_docstring:
                in_docstring = True
            else:
                in_docstring = False
                insert_pos = i + 1
                continue
        
        # Если нашли import и не в docstring
        if not in_docstring and (stripped.startswith('import ') or stripped.startswith('from ')):
            insert_pos = i
            break
    
    # Код для подавления предупреждений
    suppression_code = """
import os
import warnings

# Подавление предупреждений OpenCV о TIFF тегах
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
warnings.filterwarnings('ignore')
"""
    
    # Вставляем код
    lines.insert(insert_pos, suppression_code)
    new_content = '\n'.join(lines)
    
    # Сохраняем
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True, "Исправлено"


def main():
    """
    Главная функция
    """
    
    print("="*80)
    print(" "*15 + "🔇 ОТКЛЮЧЕНИЕ ПРЕДУПРЕЖДЕНИЙ OPENCV")
    print("="*80)
    
    print("\nЭтот скрипт добавит код для подавления предупреждений OpenCV")
    print("о неизвестных TIFF тегах во все файлы проекта.")
    
    print("\n📋 Будут исправлены файлы:")
    
    files_to_fix = [
        'add_masks_to_spacenet.py',
        'train_segmentation.py',
        'demo_pipeline.py',
        'inference.py',
        'create_local_data.py',
        'prepare_data.py',
        'check_data.py',
        'fix_spacenet_masks.py',
    ]
    
    for f in files_to_fix:
        print(f"   • {f}")
    
    response = input("\n❓ Продолжить? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y', 'да']:
        print("\n❌ Отменено.")
        return
    
    print("\n🔧 Применение исправлений...")
    print("-"*80)
    
    results = []
    
    for filename in files_to_fix:
        file_path = Path(filename)
        
        if not file_path.exists():
            results.append((filename, "⚠️  Не найден"))
            continue
        
        try:
            fixed, status = add_warning_suppression(file_path)
            
            if fixed:
                results.append((filename, "✅ Исправлено"))
            else:
                results.append((filename, f"⏭️  {status}"))
        
        except Exception as e:
            results.append((filename, f"❌ Ошибка: {e}"))
    
    # Вывод результатов
    print()
    for filename, status in results:
        print(f"   {filename:<35} {status}")
    
    print("\n" + "="*80)
    print("✅ ГОТОВО!")
    print("="*80)
    
    print("\n🔇 Предупреждения OpenCV больше не будут отображаться.")
    print("\nТеперь при запуске скриптов вы увидите только важные сообщения,")
    print("без множества строк 'WARN: TIFFReadDirectory: Unknown field...'")


if __name__ == '__main__':
    main()
