"""
Патч для исправления ошибки с cv2.circle
Запустите этот файл, если получаете ошибку:
"cv2.error: OpenCV(4.13.0) :-1: error: (-5:Bad argument) in function 'circle'"
"""

import os
import sys

def apply_patch():
    """Применяет исправление к файлам проекта"""
    
    print("="*70)
    print("ПРИМЕНЕНИЕ ПАТЧА ДЛЯ ИСПРАВЛЕНИЯ ОШИБКИ cv2.circle")
    print("="*70)
    
    # Исправление для train_segmentation.py
    train_file = "train_segmentation.py"
    
    if not os.path.exists(train_file):
        print(f"✗ Файл {train_file} не найден!")
        return False
    
    print(f"\n🔧 Исправление {train_file}...")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Старый код
    old_code = """        # Класс 3 - вода (круги)
        for _ in range(2):
            x, y = np.random.randint(20, self.img_size-20, 2)
            radius = np.random.randint(15, 40)
            cv2.circle(mask, (x, y), radius, 3, -1)"""
    
    # Новый код
    new_code = """        # Класс 3 - вода (круги)
        for _ in range(2):
            x, y = np.random.randint(20, self.img_size-20, 2)
            radius = np.random.randint(15, 40)
            # Создаём временную маску для круга
            temp_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.circle(temp_mask, (x, y), radius, 1, -1)
            mask[temp_mask == 1] = 3"""
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        with open(train_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ {train_file} исправлен")
    else:
        print(f"⚠ {train_file} уже исправлен или имеет другую структуру")
    
    # Исправление для prepare_data.py (если есть)
    prepare_file = "prepare_data.py"
    
    if os.path.exists(prepare_file):
        print(f"\n🔧 Исправление {prepare_file}...")
        
        with open(prepare_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        old_code2 = """        # Класс 3 - вода (синие области)
        for _ in range(3):
            x, y = np.random.randint(30, img_size-30, 2)
            radius = np.random.randint(20, 60)
            cv2.circle(mask, (x, y), radius, 3, -1)
            cv2.circle(image, (x, y), radius, (50, 100, 200), -1)"""
        
        new_code2 = """        # Класс 3 - вода (синие области)
        for _ in range(3):
            x, y = np.random.randint(30, img_size-30, 2)
            radius = np.random.randint(20, 60)
            # Создаём временные маски
            temp_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.circle(temp_mask, (x, y), radius, 1, -1)
            mask[temp_mask == 1] = 3
            # Рисуем на изображении
            cv2.circle(image, (x, y), radius, (50, 100, 200), -1)"""
        
        if old_code2 in content:
            content = content.replace(old_code2, new_code2)
            with open(prepare_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ {prepare_file} исправлен")
        else:
            print(f"⚠ {prepare_file} уже исправлен или имеет другую структуру")
    
    print("\n" + "="*70)
    print("✅ ПАТЧ УСПЕШНО ПРИМЕНЁН!")
    print("="*70)
    print("\nТеперь вы можете запустить:")
    print("  python demo_pipeline.py")
    print("  python train_segmentation.py")
    
    return True

if __name__ == '__main__':
    apply_patch()
