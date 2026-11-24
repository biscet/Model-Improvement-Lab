# Архитектура Model Improvement Lab

Проект разделён на переносимую оркестрацию и необязательный ML backend.
Демо, подготовка плана и просмотр статуса работают без PyTorch, CUDA и сети.
Настоящее обучение требует окружения из основной инструкции.

## Карта модулей

```text
coach.py                      общая точка запуска CLI
scripts/
  commands/                   аргументы и запуск пользовательских команд
  multimodal/                 зрение, речь, музыка и аудио/текстовые векторы
    configuration.py          рецепты и проверка локальных датасетов
    models.py                 загрузка моделей, обработка медиа и инференс
    training.py               LoRA/проекции, лимит времени и продолжение обучения
    artifacts.py              пакеты HF с лицензиями и происхождением весов
  preparation/                подготовка входов и неизменяемого плана
    validation.py             отдельные проверки полей записи
    records.py                нормализация одной записи
    inputs.py                 загрузка splits, изоляция и выбор holdout
    artifacts.py              запись входов и bounded shards
    manifest.py               хеши файлов и описание плана
    pipeline.py               последовательность подготовки run
  preflight_checks/           проверки перед обучением
    artifacts.py              конфигурация, модели, исходники и бинарники
    inputs.py                 shards, holdout и отчёт токенизатора
    resources.py              расчёт свободного места
    environment.py            версии библиотек training Python
    pipeline.py               последовательность preflight
  mentoring/                  работа с ментором
    review.py                 явный сценарий обзора evidence
    credentials.py            загрузка приватного ключа
    decisions.py              схема решения и изменение learning rate
    protocol.py               JSON-запрос и разбор ответа провайдера
    client.py                 HTTP и повторы временных ошибок
    storage.py                кеш решений, архив и учёт запросов
    strategy.py               однократное применение стратегии
  orchestration/              управление полным run
    controller.py             старт, эпохи, benchmark и финальное состояние
    session.py                run-state, сигналы и окружение процессов
    shards.py                 capture, обучение и фиксация одного shard
    runtime.py                процессы, features и восстановление сервиса
  training/                   необязательный ML backend
    pipeline.py               resume, обучение и завершение shard
    backend.py                создание модели, objective и optimizer
    engine.py                 один batch и полный проход по shard
    data.py                   samples и ограниченные срезы memmap
    objective.py              подготовка batch, loss и валидация
    position_loss.py          взвешивание позиции по D-PACE
    eagle3.py                 офлайн-этапы EAGLE-3 через SpecForge и llama.cpp
    checkpoints.py            восстановление, выбор лучших весов и commit
    reporting.py              progress events и итоговый JSON
    model_io.py               загрузка target/draft и экспорт GGUF
    gguf.py                   имена тензоров и деквантизация
    amp_step.py               повтор batch при AMP overflow
    checkpoint_seed.py        проверка checkpoint другого плана
  benchmarking/               измерения контрольного serving
    server.py                 процесс inference и варианты команды
    measurements.py           два прохода по reserved inputs
    summary.py                сравнение скорости и совпадения ответов
  coach_core/                 стандартная библиотека без ML-зависимостей
    configuration.py          конфигурация и разрешение путей
    files.py                  JSON/JSONL, SHA-256 и приватные права
    planning.py               бюджет shards и изоляция источников
    metrics.py                агрегация метрик и числовые проверки
    commands.py               аргументы дочерних процессов
    reports.py                evidence для ментора
    constants/                общие значения и фиксированный рецепт
    types/                    общие определения типов и состояния этапов
source/                       DFlash под лицензией MIT
  zmodel.py                   модели и speculative generation
  layers.py                   attention, decoder, convolution и selector
  model_utils.py             конфигурация, признаки, маски и rotary embedding
  sampling.py                 распределения и rejection sampling
capture/                      нативные capture и tokenizer
  capture.cpp                 обработка записей и запуск capture
  feature_capture.cpp         decode, генерация и сериализация features
  capture_io.cpp              окружение, paths, повторное использование и JSON
  tokenize_prompts.cpp        токенизация записей и сводка
  constants/                  фиксированные параметры захвата
  types/                      общие структуры и типы C++
patches/eagle3/               патчи к совместимым версиям SpecForge и llama.cpp
tests/                        CPU-тесты и необязательные ML-тесты
```

`coach_core` не импортирует команды или ML backend. CLI вызывает сценарии
соответствующих доменов. Все команды запускаются через `python3 coach.py …`;
дочерний trainer использует `coach.py train-shard`. Установка проекта как пакета
и изменения `sys.path` приложению не нужны. Существующие команды `coach.py`
сохранены. Прежние отдельные файлы из корня `scripts/` перенесены по доменам.

Общие определения типов хранятся в `types`, константы — в `constants`.
Папка `types` вложена в `coach_core`, чтобы не затенять стандартный модуль
Python `types` при запуске скриптов.

## Контракты, которые нужно сохранять

- Пути разрешаются относительно JSON-конфигурации. Форматы конфигурации,
  JSONL, features и checkpoint DFlash2 сохраняют совместимость. Функция потерь
  входит в неизменяемый план и записывается в checkpoint.
- `prepare` фиксирует хеши всех Python-модулей в `scripts/` и `source/`,
  включая вложенные пакеты. Изменение исходников требует нового frozen plan;
  обходить проверку существующего плана нельзя.
- После обучения shard сначала сохраняются артефакты trainer, затем получается
  решение ментора и только после этого фиксируется покрытие в run-state.
  Повторный запуск trainer распознаёт уже завершённый stage.
- Последние FP32 master weights и лучшие FP16 weights остаются разными
  состояниями. Снимки не должны ссылаться на живые параметры модели.
- Порядок параметров, имена ключей `state_dict`, отображение в GGUF,
  seed и uniform objective сохраняются. D-PACE меняет веса ошибок только
  в обучении.
- Файл блокировки закрывается и при ошибке preflight. Restore-marker удаляется
  только после успешного запуска и health check сервиса.
- Библиотечные импорты не запускают обучение, сетевые запросы или сервисы.

## Разработка

Поддерживается Python 3.10+. Версия выпуска находится в `VERSION`, изменения
до следующего выпуска — в разделе Unreleased файла CHANGELOG.

Внутренний этап обучения запускается через `coach.py train-shard` и требует зависимостей из основной
инструкции. Переносимое ядро и offline-команды используют стандартную библиотеку.

Команда `coach.py eagle3` запускает захват, офлайн-обучение и экспорт
в отдельных процессах. Рецепт, формат признаков и checkpoint EAGLE-3
задают патчи к SpecForge и llama.cpp. Установка среды описана в
[инструкции EAGLE-3](eagle3.ru.md).

Рефакторинг сохраняет лицензию MIT у извлечённых частей DFlash.

## Мультимодальные эксперименты

Команда `coach.py multimodal` использует отдельное окружение и рецепты
из `configs/*.example.json`. Датасеты, медиа, веса и результаты остаются
в локальных каталогах. Форматы и команды описаны в [руководстве](multimodal.md).
