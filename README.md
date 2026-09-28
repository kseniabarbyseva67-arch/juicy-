# Falcon Vehicle ReID Service

Рабочий MVP сервиса формирования цифрового признака транспортного средства и поиска похожих автомобилей по изображению **без распознавания госномера**.

## Что реализовано

- FastAPI backend + автоматически генерируемая OpenAPI/Swagger документация (`/docs`).
- Приём изображения и BBox, базовая валидация входа.
- Формирование фиксированного L2-нормированного эмбеддинга размерности 512.
- Галерея в реляционной SQLite БД: эмбеддинги + метаданные.
- Косинусный поиск top-N и режим отказа по настраиваемому порогу.
- Тонкий web-клиент в браузере.
- Dockerfile и `docker-compose.yml` для запуска одной командой.
- Unit-тесты.
- Скрипт подготовки `embeddings.npy`, `submission.csv`, `candidates.csv`.

> Текущий embedder — полностью офлайн baseline на визуальных признаках (цвет + пространственная структура + края). Архитектура отделяет embedder от API/хранилища, поэтому для соревнования его следует заменить на обученный ReID backbone (например, ConvNeXt/ViT/OSNet), сохранив API без изменений.

## Быстрый запуск

```bash
docker compose up --build
```

Открыть:
- UI: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/health

## API

### Добавить объект в галерею
`POST /api/v1/gallery/items` — multipart form:
- `image`: JPEG/PNG
- `image_id`: идентификатор исходного кадра
- `vehicle_id`: опционально
- `bbox`: JSON, например `{"x":10,"y":20,"w":300,"h":180}`

### Найти кандидатов
`POST /api/v1/query` — multipart form:
- `image`
- `bbox`
- `top_k` (по умолчанию 10)
- `threshold` (по умолчанию 0.82)

Ответ содержит `refused`, `top_score` и кандидатов. Если top-score ниже порога, список кандидатов пуст.

## Калибровка порога отказа

Значение `0.82` в MVP является стартовым техническим значением, а не утверждением о качестве. Для финального решения порог нужно выбрать на validation split по F1/TNR (или PR-кривой), сохранить процедуру подбора и обосновать её на защите.

## Переход к соревновательной модели

Рекомендуемая замена baseline embedder:
1. Crop по BBox с небольшим padding.
2. ReID backbone 256–1024d, предобученный на открытом датасете и дообученный на train split.
3. Metric learning: ArcFace/CosFace/Triplet + classification head.
4. Аугментации: viewpoint, illumination, blur, occlusion, weather-like transforms.
5. L2 normalization + cosine distance.
6. Hard-negative mining и валидация open-set refusal.
7. При галерее ~10^6 объектов заменить exact NumPy search на FAISS/HNSW/Milvus/pgvector.

## Тесты

```bash
pytest -q
```

## Переменные окружения

- `REFUSAL_THRESHOLD=0.82`
- `TOP_K=10`
- `MAX_IMAGE_MB=20`
- `DATA_DIR=/app/data`

## Ограничения MVP

- Baseline предназначен для демонстрации полного сервиса, API и продуктового сценария, а не для достижения высокого mAP на закрытом ReID-тесте.
- Временная SQLite + exact NumPy retrieval архитектура подходит для прототипа и небольших галерей; для 10^6 элементов нужен ANN индекс.
- Госномер не распознаётся и не используется.
