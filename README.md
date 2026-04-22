# License Plate Detection (YOLO)
## Автор

- **ФИО:** Ruslan Agafonov & Angelina Chernikova
- **Группа:** 972401

## Возможности

- Обучение модели на своем датасете (`train`)
- Оценка качества на размеченном наборе (`evaluate`)
- Обработка видеофайла с сохранением результата (`video`)
- Логирование в `./data/log_file.log`

## Структура проекта

- `main.py` — CLI entrypoint
- `src/model_impl.py` — класс `My_LicensePlate_Model`
- `src/train.py` — обучение
- `src/evaluate.py` — валидация и метрики
- `src/video_mode.py` — обработка видео
- `Dockerfile`, `docker-compose.yaml` — контейнеризация

## Требования

- Python 3.10+
- (Опционально) CUDA GPU для ускорения обучения/инференса

Установка зависимостей:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Как запускать

### 1) Обучение

```bash
python3 main.py train
```

### 2) Оценка модели

```bash
python3 main.py evaluate \
  --model runs/detect/detect/train/weights/best.pt \
  --images data/test/images \
  --labels data/test/labels \
  --iou 0.5 \
  --conf 0.5 \
  --plot \
  --output evaluation_results.jsons
```

### 3) Обработка видео

```bash
python3 main.py video \
  --input test.mp4 \
  --output output.mp4 \
  --model runs/detect/detect/train/weights/best.pt \
  --conf 0.5 \
  --show
```

## Docker

Сборка:

```bash
docker compose build
```

Показать help:

```bash
docker compose up
```

Запустить конкретную команду:

```bash
docker compose run --rm app python3 main.py info
docker compose run --rm app python3 main.py video -i test.mp4 -o output.mp4
```

Логи доступны в контейнере и на хосте по пути:

```text
./data/log_file.log
```