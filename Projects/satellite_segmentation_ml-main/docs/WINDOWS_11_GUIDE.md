# 🪟 Инструкция для Windows 11

## 🎯 Быстрый старт (3 шага)

### Шаг 1: Скачайте данные SpaceNet

```powershell
python download_spacenet_windows.py
```

Скрипт автоматически:
- ✅ Скачает 3 датасета SpaceNet (~4 ГБ)
- ✅ Распакует архивы
- ✅ Организует в папки `data/images` и `data/masks`
- ✅ Очистит временные файлы

### Шаг 2: Исправьте ошибку OpenCV (если нужно)

```powershell
python fix_cv2_error.py
```

### Шаг 3: Запустите обучение

```powershell
# Быстрая демонстрация (3 эпохи, ~5-10 минут)
python demo_pipeline.py

# Или полное обучение (20 эпох, ~30-60 минут)
python train_segmentation.py
```

---

## 📋 Детальная инструкция

### 1️⃣ Подготовка окружения

```powershell
# Создайте виртуальное окружение (если ещё не создали)
python -m venv venv

# Активируйте его
.\venv\Scripts\Activate.ps1

# Установите зависимости
pip install -r requirements.txt
```

### 2️⃣ Скачивание данных SpaceNet

#### Вариант А: Автоматически (Рекомендуется)

```powershell
python download_spacenet_windows.py
```

**Что будет скачано:**
- Germany Training Data (~1.5 ГБ)
- Louisiana-East Training Data (~1.5 ГБ)
- Louisiana-West Test Data (~1.0 ГБ)

**Время:** 10-30 минут (зависит от скорости интернета)

**Результат:**
```
C:\Users\Dmitry\Desktop\neuroLabs\lab1\
├── data\
│   ├── images\     (реальные спутниковые снимки)
│   └── masks\      (маски сегментации)
```

#### Вариант Б: Вручную через браузер

Если автоматическое скачивание не работает:

1. **Скачайте архивы:**
   - [Germany Training](https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Germany_Training_Public.tar.gz)
   - [Louisiana-East Training](https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Louisiana-East_Training_Public.tar.gz)
   - [Louisiana-West Test](https://spacenet-dataset.s3.amazonaws.com/spacenet/SN8_floods/tarballs/Louisiana-West_Test_Public.tar.gz)

2. **Распакуйте с помощью 7-Zip или WinRAR:**
   - Установите [7-Zip](https://www.7-zip.org/)
   - Правый клик → 7-Zip → Extract Here

3. **Организуйте файлы:**
   ```powershell
   python prepare_data.py
   # Выберите опцию 1
   ```

#### Вариант В: Синтетические данные (быстро)

Если не хотите скачивать 4 ГБ:

```powershell
python create_local_data.py
```

Создаст 500 синтетических примеров (~500 МБ) за 2-3 минуты.

### 3️⃣ Проверка данных

```powershell
# Проверьте, что данные на месте
dir data\images
dir data\masks

# Должно быть файлов:
# - Реальные данные: 500-2000 файлов
# - Синтетические: 500 файлов
```

### 4️⃣ Запуск обучения

#### Быстрая демонстрация (3 эпохи)

```powershell
python demo_pipeline.py
```

**Результат:**
- `outputs/demo/training_curves.png` - графики обучения
- `outputs/demo/predictions.png` - примеры предсказаний
- `outputs/demo/metrics_by_class.png` - метрики
- `models/demo_model.pth` - обученная модель

#### Полное обучение (20 эпох)

```powershell
python train_segmentation.py
```

**Результат:**
- `outputs/predictions.png` - визуализация
- `outputs/training_history.png` - графики
- `outputs/training_history.json` - метрики
- `models/best_unet_model.pth` - модель

### 5️⃣ Инференс на новых изображениях

```powershell
# Используйте обученную модель для предсказаний
python inference.py --image путь\к\изображению.png
```

---

## 🔧 Решение проблем

### Проблема 1: Ошибка cv2.circle

```
cv2.error: OpenCV(4.13.0) :-1: error: (-5:Bad argument)
```

**Решение:**
```powershell
python fix_cv2_error.py
```

### Проблема 2: Скачивание не работает

**Вариант 1:** Используйте браузер (см. Вариант Б выше)

**Вариант 2:** Синтетические данные
```powershell
python create_local_data.py
```

### Проблема 3: Мало места на диске

**Минимальные требования:**
- Реальные данные: ~10 ГБ
- Синтетические: ~1 ГБ

**Очистка после скачивания:**
Скрипт `download_spacenet_windows.py` предложит удалить временные файлы (освободит ~8 ГБ).

### Проблема 4: Медленная работа на CPU

**Решение 1:** Уменьшите параметры

Откройте `train_segmentation.py` или `demo_pipeline.py` и измените:

```python
IMG_SIZE = 128        # вместо 256
BATCH_SIZE = 4        # вместо 8
NUM_EPOCHS = 5        # вместо 20
```

**Решение 2:** Используйте GPU

Установите PyTorch с поддержкой CUDA:
```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Проблема 5: ImportError или ModuleNotFoundError

```powershell
# Переустановите зависимости
pip install -r requirements.txt --upgrade
```

---

## 📊 Ожидаемые результаты

### После обучения на синтетических данных:
```
Mean IoU:  0.75-0.85
Mean Dice: 0.80-0.88

IoU по классам:
  Фон:     0.85-0.90
  Здания:  0.70-0.80
  Дороги:  0.75-0.85
  Вода:    0.65-0.75
  Поля:    0.70-0.80
```

### После обучения на реальных данных SpaceNet:
```
Mean IoU:  0.65-0.75
Mean Dice: 0.70-0.80

(Реальные данные сложнее, поэтому метрики ниже)
```

---

## 🎓 Полезные команды

```powershell
# Просмотр структуры проекта
tree /F /A

# Проверка размера данных
dir data /s

# Активация виртуального окружения
.\venv\Scripts\Activate.ps1

# Деактивация
deactivate

# Проверка установленных пакетов
pip list

# Запуск Jupyter Notebook
jupyter notebook segmentation_notebook.ipynb
```

---

## 📁 Итоговая структура проекта

```
C:\Users\Dmitry\Desktop\neuroLabs\lab1\
├── venv\                           # Виртуальное окружение
├── data\                           # Данные (создаётся автоматически)
│   ├── images\
│   ├── masks\
│   └── README.txt
├── data_raw\                       # Скачанные архивы (можно удалить)
├── data_extracted\                 # Распакованные файлы (можно удалить)
├── models\                         # Сохранённые модели
│   ├── best_unet_model.pth
│   └── demo_model.pth
├── outputs\                        # Результаты обучения
│   ├── predictions.png
│   ├── training_history.png
│   └── demo\
├── train_segmentation.py           # Основной скрипт
├── demo_pipeline.py                # Быстрая демонстрация
├── inference.py                    # Инференс
├── download_spacenet_windows.py    # Скачивание данных ⭐
├── create_local_data.py            # Синтетические данные
├── fix_cv2_error.py               # Исправление ошибки
├── prepare_data.py                # Подготовка данных
├── alternative_models.py          # Другие архитектуры
├── requirements.txt               # Зависимости
└── README.md                      # Документация
```

---

## ⏱️ Время выполнения (примерно)

| Задача | CPU | GPU (NVIDIA) |
|--------|-----|--------------|
| Скачивание данных | 10-30 мин | 10-30 мин |
| Создание синтетических данных | 2-3 мин | 2-3 мин |
| Обучение (3 эпохи, 200 примеров) | 5-10 мин | 1-2 мин |
| Обучение (20 эпох, 800 примеров) | 40-60 мин | 5-10 мин |
| Инференс (1 изображение) | <1 сек | <1 сек |

---

## ✅ Чек-лист запуска

- [ ] Установлен Python 3.8+
- [ ] Создано виртуальное окружение
- [ ] Установлены зависимости (`pip install -r requirements.txt`)
- [ ] Скачаны данные (`python download_spacenet_windows.py`)
- [ ] Исправлена ошибка OpenCV (`python fix_cv2_error.py`)
- [ ] Запущено обучение (`python demo_pipeline.py`)
- [ ] Проверены результаты (папка `outputs/`)

---

## 🆘 Нужна помощь?

1. **Проверьте логи** - все ошибки выводятся в консоль
2. **Проверьте README.md** - подробная документация
3. **Используйте синтетические данные** - быстрее и проще для начала

---

**Удачи в обучении модели! 🚀**
