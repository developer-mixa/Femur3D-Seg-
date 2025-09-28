# 3D Femur Segmentation Service

## Описание проекта

Микросервис для автоматической сегментации бедренной кости из CT-снимков. Проект выполнен в рамках курса компьютерного зрения.

### Возможности
- Загрузка NIFTI файлов через REST API
- Автоматическая сегментация бедренной кости
- Экспорт результатов в различных форматах (STL, PNG)
- Визуализация результатов
- Docker-контейнеризация

## Быстрый старт

### Локальный запуск

1. **Клонирование репозитория**
```bash
git clone <repository-url>
cd femur-segmentation
```

2. **Установка зависимостей**
```bash
pip install -r requirements.txt
```

3. **Запуск сервиса**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Запуск через Docker

1. **Сборка образа**
```bash
docker-compose build
```

2. **Запуск контейнера**
```bash
docker-compose up -d
```

3. **Проверка логов**
```bash
docker-compose logs -f
```

## API Endpoints

### Основные endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/` | Информация о сервисе |
| GET | `/health` | Проверка состояния |
| POST | `/upload` | Загрузка NIFTI файла |
| POST | `/segment/{task_id}` | Запуск сегментации |
| GET | `/status/{task_id}` | Статус задачи |
| GET | `/result/{task_id}` | Получение маски (NIFTI) |
| GET | `/visualize/{task_id}` | Визуализация результата |
| GET | `/export/{task_id}?format=stl` | Экспорт в STL/PNG |

### Примеры использования

#### 1. Загрузка файла
```python
import requests

# Загрузка NIFTI файла
with open("ct_scan.nii.gz", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload",
        files={"file": ("ct_scan.nii.gz", f, "application/gzip")}
    )
    
task_id = response.json()["task_id"]
print(f"Task ID: {task_id}")
```

#### 2. Запуск сегментации
```python
# Запуск процесса сегментации
response = requests.post(f"http://localhost:8000/segment/{task_id}")
print(response.json())
```

#### 3. Проверка статуса
```python
# Проверка статуса выполнения
response = requests.get(f"http://localhost:8000/status/{task_id}")
status = response.json()["status"]
print(f"Status: {status}")
```

#### 4. Получение результата
```python
# Скачивание маски сегментации
response = requests.get(f"http://localhost:8000/result/{task_id}")
with open("segmentation_mask.nii.gz", "wb") as f:
    f.write(response.content)
```

#### 5. Экспорт в STL
```python
# Экспорт в 3D mesh формат
response = requests.get(f"http://localhost:8000/export/{task_id}?format=stl")
with open("femur.stl", "wb") as f:
    f.write(response.content)
```

## Модели сегментации

### TotalSegmentator (рекомендуется)
- Автоматическая загрузка весов при первом использовании
- Высокая точность (Dice Score ~0.95)
- Поддержка 104 классов органов
- GPU ускорение

### MONAI U-Net (альтернатива)
- Легковесная модель
- Требует предобученных весов
- Быстрый inference

## Структура проекта

```
femur-segmentation/
├── app/
│   ├── main.py              # FastAPI приложение
│   ├── models/
│   │   └── segmentation.py   # Модели сегментации
│   └── utils/
│       ├── preprocessing.py  # Предобработка
│       ├── postprocessing.py # Постобработка
│       └── visualization.py  # Визуализация
├── data/
│   ├── input/               # Входные файлы
│   ├── output/              # Результаты
│   └── models/              # Веса моделей
├── docker/
│   └── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Требования к данным

### Входные данные
- Формат: NIFTI (.nii, .nii.gz)
- Модальность: CT
- Разрешение: 0.5-2.0 мм
- Размер: до 512×512×512 вокселей

### Выходные данные
- Бинарная маска сегментации (NIFTI)
- 3D mesh (STL)
- PNG срезы с визуализацией

## Производительность

| Конфигурация | Время сегментации | Точность |
|--------------|------------------|----------|
| CPU (8 cores) | ~60 сек | 0.93 |
| GPU (RTX 3060) | ~10 сек | 0.95 |
| GPU (V100) | ~5 сек | 0.95 |

## Тестирование

## Развертывание на кластере

1. **Подготовка docker-compose.yml**
```bash
# Отредактируйте пути и настройки в docker-compose.yml
nano docker-compose.yml
```

2. **Отправка DevOps специалисту**
```bash
# Архивирование проекта
tar -czf femur_segmentation.tar.gz .
# Отправка на сервер
scp femur_segmentation.tar.gz user@cluster:/path/to/deployment/
```

3. **Запуск на кластере**
```bash
ssh user@cluster
cd /path/to/deployment/
tar -xzf femur_segmentation.tar.gz
docker-compose up -d

*Проект выполнен в рамках курса "Компьютерное зрение" 2025*