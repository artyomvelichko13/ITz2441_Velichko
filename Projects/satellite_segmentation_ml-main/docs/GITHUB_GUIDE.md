# 📤 Инструкция по публикации проекта на GitHub

## Шаг 1: Создание репозитория на GitHub

1. Откройте [GitHub.com](https://github.com) и войдите в свой аккаунт
2. Нажмите на "+" в правом верхнем углу → "New repository"
3. Заполните информацию:
   - **Repository name:** `satellite-segmentation`
   - **Description:** "Semantic segmentation of satellite imagery using U-Net"
   - **Public** (обязательно для преподавателя!)
   - ❌ НЕ ставьте галочку "Initialize this repository with a README"
4. Нажмите "Create repository"

## Шаг 2: Инициализация локального репозитория

Откройте PowerShell в папке проекта:

```powershell
cd C:\Users\Dmitry\Desktop\neuroLabs\lab1
```

### Инициализируйте Git:

```powershell
git init
git add .
git commit -m "Initial commit: U-Net satellite segmentation project"
```

### Свяжите с GitHub:

Замените `your-username` на ваш GitHub username:

```powershell
git remote add origin https://github.com/your-username/satellite-segmentation.git
git branch -M main
git push -u origin main
```

### Если Git попросит аутентификацию:

- Используйте GitHub Personal Access Token
- Или настройте SSH ключ

## Шаг 3: Проверка структуры

После push, убедитесь, что на GitHub есть:

```
satellite-segmentation/
├── README.md               ✅ Главное описание
├── requirements.txt        ✅ Зависимости
├── LICENSE                 ✅ Лицензия
├── .gitignore             ✅ Игнорируемые файлы
├── src/                   ✅ Исходный код
│   ├── __init__.py
│   ├── train_segmentation.py
│   ├── demo_pipeline.py
│   ├── inference.py
│   └── ...
├── notebooks/             ✅ Jupyter notebooks
│   └── segmentation_notebook.ipynb
├── reports/               ✅ Результаты
│   ├── README.md
│   ├── figures/
│   └── metrics/
├── models/                ✅ Папка для моделей (пустая)
└── docs/                  ✅ Документация
```

## Шаг 4: Добавление результатов обучения

### После запуска обучения:

```powershell
# Запустите обучение
python src/demo_pipeline.py

# Скопируйте результаты в reports/
copy outputs\demo\*.png reports\figures\
copy outputs\*.json reports\metrics\

# Добавьте в Git
git add reports/figures reports/metrics
git commit -m "Add training results and visualizations"
git push
```

## Шаг 5: Обновление README с реальными результатами

Отредактируйте `README.md` и добавьте ссылки на изображения:

```markdown
![Training curves](reports/figures/training_history.png)
![Predictions](reports/figures/predictions.png)
```

Затем:

```powershell
git add README.md
git commit -m "Update README with results"
git push
```

## Шаг 6: Отправка ссылки преподавателю

Скопируйте ссылку на репозиторий:

```
https://github.com/your-username/satellite-segmentation
```

Отправьте преподавателю!

## Дополнительные команды Git

### Проверить статус:
```powershell
git status
```

### Добавить конкретный файл:
```powershell
git add src/train_segmentation.py
```

### Откатить изменения:
```powershell
git checkout -- filename.py
```

### Обновить удалённый репозиторий:
```powershell
git pull
```

## Что НЕ должно быть в репозитории

❌ Папка `data/` (слишком большая)
❌ Папка `venv/` (виртуальное окружение)
❌ Файлы `.pth` (модели слишком большие)
❌ Папка `__pycache__/`

Всё это уже в `.gitignore`!

## Troubleshooting

### Проблема: "remote origin already exists"

```powershell
git remote remove origin
git remote add origin https://github.com/your-username/satellite-segmentation.git
```

### Проблема: "Permission denied"

Используйте Personal Access Token вместо пароля:
1. GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Выберите `repo` scope
4. Используйте токен вместо пароля

### Проблема: Большие файлы

Если случайно добавили большие файлы:

```powershell
git rm --cached -r data/
git rm --cached models/*.pth
git commit -m "Remove large files"
git push
```

## Полезные ссылки

- [GitHub Desktop](https://desktop.github.com/) - GUI для Git
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
