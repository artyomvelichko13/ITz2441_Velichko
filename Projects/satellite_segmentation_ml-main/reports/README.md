# 📊 Reports and Results

Эта папка содержит результаты обучения модели, визуализации и метрики.

## Структура

```
reports/
├── figures/              # Графики и изображения
│   ├── training_history.png
│   ├── predictions.png
│   ├── metrics_by_class.png
│   └── example_prediction.png
└── metrics/              # Метрики в JSON/CSV
    └── training_history.json
```

## Содержимое

### figures/

**training_history.png** - График обучения с динамикой loss, IoU и Dice

**predictions.png** - Примеры предсказаний модели на валидационной выборке

**metrics_by_class.png** - Гистограммы метрик IoU и Dice по классам

**example_prediction.png** - Пример сегментации с оригиналом, ground truth и предсказанием

### metrics/

**training_history.json** - Полная история обучения в формате JSON:
- train_loss по эпохам
- val_loss по эпохам
- mean_iou по эпохам
- mean_dice по эпохам
- class_iou по эпохам для каждого класса
- class_dice по эпохам для каждого класса

## Как сгенерировать

Результаты генерируются автоматически при запуске:

```bash
python src/train_segmentation.py
# или
python src/demo_pipeline.py
```

Файлы сохраняются в:
- `outputs/` - при запуске train_segmentation.py
- `outputs/demo/` - при запуске demo_pipeline.py

Скопируйте нужные файлы в `reports/figures/` для включения в репозиторий.
