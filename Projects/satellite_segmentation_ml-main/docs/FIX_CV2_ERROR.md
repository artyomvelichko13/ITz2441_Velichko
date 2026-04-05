# 🔧 ИСПРАВЛЕНИЕ ОШИБКИ cv2.circle

## Проблема

При запуске `demo_pipeline.py` или `train_segmentation.py` возникает ошибка:

```
cv2.error: OpenCV(4.13.0) :-1: error: (-5:Bad argument) in function 'circle'
> Overload resolution failed:
>  - Layout of the output array img is incompatible with cv::Mat
```

## Причина

OpenCV требует, чтобы массив для `cv2.circle()` был типа `uint8`, а не `int64`.

## Решение

### Вариант 1: Автоматическое исправление (Рекомендуется)

Запустите патч-файл:

```bash
python fix_cv2_error.py
```

Этот скрипт автоматически исправит все проблемные файлы.

### Вариант 2: Ручное исправление

#### В файле `train_segmentation.py`

Найдите метод `__getitem__` класса `SyntheticDataset` (примерно строка 190-210):

**Было:**
```python
# Класс 3 - вода (круги)
for _ in range(2):
    x, y = np.random.randint(20, self.img_size-20, 2)
    radius = np.random.randint(15, 40)
    cv2.circle(mask, (x, y), radius, 3, -1)
```

**Должно быть:**
```python
# Класс 3 - вода (круги)
for _ in range(2):
    x, y = np.random.randint(20, self.img_size-20, 2)
    radius = np.random.randint(15, 40)
    # Создаём временную маску для круга
    temp_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.circle(temp_mask, (x, y), radius, 1, -1)
    mask[temp_mask == 1] = 3
```

#### В файле `prepare_data.py`

Найдите функцию `create_sample_data` (примерно строка 220-240):

**Было:**
```python
# Класс 3 - вода (синие области)
for _ in range(3):
    x, y = np.random.randint(30, img_size-30, 2)
    radius = np.random.randint(20, 60)
    cv2.circle(mask, (x, y), radius, 3, -1)
    cv2.circle(image, (x, y), radius, (50, 100, 200), -1)
```

**Должно быть:**
```python
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
```

### Вариант 3: Изменение типа маски

Альтернативно, можно изменить тип маски при создании:

**В `train_segmentation.py`, класс `SyntheticDataset`, метод `__getitem__`:**

```python
# Вместо:
mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)

# Используйте:
mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
```

И в конце метода:

```python
# Вместо:
mask = torch.from_numpy(mask).long()

# Используйте:
mask = torch.from_numpy(mask.astype(np.int64)).long()
```

## Проверка

После применения исправления запустите:

```bash
python demo_pipeline.py
```

Если всё работает корректно, вы увидите:

```
======================================================================
               🛰️  ДЕМОНСТРАЦИЯ ПОЛНОГО ПАЙПЛАЙНА
======================================================================

📋 ПАРАМЕТРЫ:
   • Устройство: cpu
   • Размер изображения: 256x256
   ...

======================================================================
ШАГ 1: СОЗДАНИЕ ДАТАСЕТА
======================================================================
✓ Обучающая выборка: 200 примеров
✓ Валидационная выборка: 50 примеров

======================================================================
ШАГ 2: СОЗДАНИЕ МОДЕЛИ U-NET
======================================================================
✓ Модель создана
✓ Всего параметров: 31,043,781

======================================================================
ШАГ 3: ОБУЧЕНИЕ МОДЕЛИ
======================================================================

Эпоха 1/3:
   • Train Loss: ...
   • Val Loss: ...
   • Val IoU: ...
```

## Дополнительная информация

Эта ошибка возникает из-за несовместимости типов данных между NumPy и OpenCV в новых версиях OpenCV (4.13+). Исправление создаёт промежуточную маску типа `uint8`, которую OpenCV может обработать, а затем копирует значения в основную маску.

## Если проблема не решена

1. Проверьте версию OpenCV:
   ```bash
   python -c "import cv2; print(cv2.__version__)"
   ```

2. Попробуйте переустановить OpenCV:
   ```bash
   pip uninstall opencv-python
   pip install opencv-python==4.8.1.78
   ```

3. Или используйте альтернативный подход без `cv2.circle()`:
   ```python
   # Создание круга через numpy
   yy, xx = np.ogrid[:self.img_size, :self.img_size]
   circle = (xx - x)**2 + (yy - y)**2 <= radius**2
   mask[circle] = 3
   ```
