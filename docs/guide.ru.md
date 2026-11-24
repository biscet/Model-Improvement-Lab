# Установка и запуск Model Improvement Lab 3.0.0

[English version](guide.en.md) | [Описание проекта](../README.ru.md)

Инструкция описывает готовый рецепт DFlash2 для локальной языковой модели.
Команда `coach.py run` запускает захват признаков целевой модели.

## 1. Где запускать

На macOS и Linux можно проверять исходники, запускать демо и читать статус.
Для захвата признаков и обучения нужен Linux с NVIDIA CUDA и совместимые файлы
моделей. Копия проекта на Mac сама по себе не является GPU-сервисом.

В примере используются две видимые видеокарты. Основной расчёт target и draft
идёт на первой. Таблица per-layer token embeddings целевой модели вынесена на
вторую. В `gpu_uuids` задайте индексы устройств или их UUID в нужном порядке.
Для одной видеокарты также измените размещение тензоров в `server_argv`.

В готовом рецепте DFlash2 заданы следующие параметры.

| Параметр | Значение |
| --- | --- |
| Слои признаков target, нумерация с единицы | 1, 6, 11, 16, 22, 27, 32, 37 |
| Ширина признаков одного слоя | 2048 |
| Размер словаря target | 248320 |
| Число слоёв transformer у draft | 6 |
| Точность обучения | FP32 master weights, FP16 autocast |
| Файлы признаков | FP16, 8 x 2048 значений на токен |

Готовый рецепт рассчитан на модель, совместимую с Qwen3.8 35B A3B.
Для другой архитектуры целевой модели подготовьте подходящие токенизатор,
захват признаков, draft и рецепт обучения.

## 2. Проверка без установки зависимостей

Из корня проекта:

```bash
python3 coach.py demo --run-dir ../model-improvement-lab-demo
python3 coach.py status --run-dir ../model-improvement-lab-demo
python3 -m unittest discover -s tests -v
python3 coach.py privacy-check
```

Демо создаёт 24 синтетических входа, три порции и искусственные решения
наставника. Оно не выполняет обучение и не отправляет запросы в сеть.
Если PyTorch не установлен, часть тестов будет пропущена. Для обычного Mac
без обучающего окружения это ожидаемо.

## 3. Окружение для Linux

Для обучения используйте Python 3.11 или 3.12. На диске должно хватать места
для моделей, признаков и атомарной замены checkpoint. Размер одной порции
признаков не равен всему необходимому месту.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Сборка PyTorch с CUDA должна подходить к драйверу и видеокарте. Проверьте её:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
```

Файл requirements фиксирует версии библиотек. Он не устанавливает драйвер,
CUDA toolkit и веса моделей.

## 4. Сборка runtime и утилит захвата

Утилиты используют расширенные API из
[mistrjirka/llama.cpp](https://github.com/mistrjirka/llama.cpp).
В обычном llama.cpp этих API может не быть. Опорная ревизия исходников:
`e62f86f96cafaa10dde027404b9dd5a45cbb6899`.

```bash
git clone https://github.com/mistrjirka/llama.cpp.git runtime
git -C runtime checkout e62f86f96cafaa10dde027404b9dd5a45cbb6899
git -C runtime apply ../capture/qwen4exp-feature-mean.patch
python -m pip install -e runtime/gguf-py
cmake -S runtime -B runtime/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build runtime/build --target llama-server -j2
cmake -S capture -B build -DLLAMA_CPP_DIR="$PWD/runtime" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target dflash-capture dflash-tokenize -j2
```

Патч усредняет hyper-connection streams при снятии признаков. Не меняйте это
определение посреди обучения. Для другой ревизии runtime может потребоваться
адаптация патча. Ошибку сборки или smoke-проверки нужно исправить до обучения,
а не обходить удалением патча.

При необходимости задайте CUDA architectures для своих GPU. Пути к библиотекам
нестандартной установки CUDA можно передать через объект `environment` в JSON.
Пути с другого компьютера копировать не нужно.

## 5. Модели и локальные настройки

```bash
mkdir -p models
cp configs/example.json configs/training.local.json
```

Положите совместимую целевую модель в `models/target.gguf`, а начальный F16 draft
в `models/draft.gguf`, либо измените пути в JSON. Trainer читает замороженные
embedding и output head из GGUF. Отдельные полные FP16/BF16-веса target ему не нужны.

Относительные пути считаются от папки JSON. Например, `../models/target.gguf`
в `configs/training.local.json` указывает на папку models в корне проекта.
Сохраните окончание `.local.json`, чтобы Git игнорировал ваши настройки.

Проверьте основные поля:

| Поле | Для чего нужно |
| --- | --- |
| `run_dir` | Новая папка эксперимента либо существующая папка для возобновления |
| `dataset` | Папка с train.jsonl, validation.jsonl и test.jsonl |
| `python` | Python обучающего окружения |
| `capture_binary`, `tokenize_binary`, `server_binary` | Собранные исполняемые файлы |
| `gpu_uuids` | Видимые устройства в нужном CUDA-порядке |
| `capture_context` | Окно для самого длинного промпта и новой генерации |
| `training_objective` | `uniform` по умолчанию или `dpace` |
| `shard_feature_budget_gib` | Верхняя оценка размера признаков одной train-порции |
| `validation_monitor_rows`, `benchmark_test_rows` | Размеры отложенных рабочих выборок |
| `max_epochs` | Предел полных проходов; null разрешает наставнику запросить ещё |
| `server_argv` | Полный список аргументов inference-сервера для сравнения |
| `production_service` | Необязательный сервис, который нужно остановить и вернуть |

Пример допускает один проход. Наставник может закончить раньше, после любой
порции. Общий лимит API-токенов не задан. Временные ошибки API повторяются;
ошибка авторизации завершает запуск. Расход и тарифы провайдера проверяйте отдельно.

Чтобы включить D-PACE, задайте `"training_objective": "dpace"` перед
`prepare`. Рецепт сглаживает вероятность правильного токена с α = 0,5,
перемножает вероятности вдоль префикса и взвешивает каждую позицию по сумме
вероятностей префиксов, в которые она входит. Градиент через веса не идёт.
Веса применяются к ошибкам токенов и к ошибкам селектора на позициях,
где правильный токен входит в top-16 draft. Итог делится на число блоков.

Для валидации и выбора checkpoint используется uniform loss. Это даёт
общую шкалу для сравнения. Функция потерь закрепляется в плане обучения.
Чтобы сменить её, подготовьте новый `run_dir`. Формула и парный эксперимент
описаны в ML-Docs. [D-PACE](https://github.com/biscet/ML-Docs/blob/main/docs/experiments/position-weighting.ru.md)

## 6. Публичные датасеты

В проекте есть примеры для двух наборов:

- [DFlash DeepSeek TraceMix](https://huggingface.co/datasets/biscet/DFlash-DeepSeek-TraceMix).
- [DFlash ChatGPT TraceMix Lite](https://huggingface.co/datasets/biscet/DFlash-ChatGPT-TraceMix-Lite).

Ревизии и пути к файлам заданы в `examples/huggingface-datasets.json`.
Для этих публичных файлов HF-токен не нужен. Загрузчик оставляет только поля
промптов. Готовые ответы и полные provenance-записи он не переносит.
Никакие файлы обратно на Hugging Face не отправляются.

Выберите один вариант и пустую папку назначения.

Основной корпус:

```bash
python coach.py download-example --example deepseek --output data/example
```

Только отдельные 272 обучающих контекста Stage-2:

```bash
python coach.py download-example --example stage2 --output data/example
```

Небольшой OpenMix:

```bash
cp configs/lite.example.json configs/training.local.json
python coach.py download-example --example lite --output data/example
```

В исходном Lite есть train и validation. Загрузчик делит validation по целым
семействам исходных файлов на новые validation и test. Записи исходного train
в test не переносятся. Маленький конфиг использует по 16 проверочных и тестовых
записей. Пример Stage-2 использует validation/test основного DeepSeek TraceMix.

Такое разделение не доказывает независимость от всей истории обучения draft.
Если история родителя доступна, проверьте известные пересечения. Условия
источников и сторонние лицензии датасетов продолжают действовать.

## 7. Свои данные

Положите UTF-8 файлы `train.jsonl`, `validation.jsonl` и `test.jsonl` в папку
из поля dataset. Одна строка содержит один готовый промпт target.
Синтетический образец полей находится в `examples/input-record.json`.

Нужны поля `id`, `project`, `source_path`, `source_sha256`, `sampling_group`,
`prompt`, `language`. Точное число токенов добавляется следующим шагом.
ID может содержать буквы, цифры, подчёркивание и дефис. Связанные файлы
и языковые варианты должны оставаться в одном split. Стабильные hash и пути
нужны для проверки известных пересечений.

Промпты заранее формируются chat template целевой модели. Публичные примеры
используют записанный в них шаблон. Пересчёт токенов не исправляет чужой template
и не делает ответы другого teacher совместимыми с target. Обучающие ответы
в любом случае заново генерирует настроенная целевая модель.

## 8. Подсчёт токенов и план

```bash
python coach.py tokenize-inputs --config configs/training.local.json
python coach.py prepare --config configs/training.local.json
python coach.py verify-tokens --config configs/training.local.json
```

`tokenize-inputs` обновляет точные длины на CPU. Для Lite этот шаг обязателен:
в исходных записях нет prompt_tokens. `prepare` проверяет границы split, делит
train на порции и считает hashes входов, кода, бинарников и моделей.
`verify-tokens` повторно проверяет уже зафиксированные входы захвата.
Эти команды не начинают GPU-работу и не вызывают DeepSeek.

После подготовки не меняйте данные, код, runtime, модели и конфиг этого запуска.
При несовпадении hash preflight остановит работу. Для другого эксперимента
используйте новую папку run_dir. Не обходите проверку правкой plan.json.

## 9. DeepSeek

Пример настроен на `deepseek-flash` через DeepSeek API. Аккаунт должен
поддерживать этот alias и поля запроса. Название модели можно изменить в конфиге.

Для запуска в терминале задайте переменную окружения `DEEPSEEK_API_KEY`.
Не записывайте значение в отслеживаемый JSON, аргумент команды или README.

Для фонового user-сервиса используйте отдельный закрытый файл вне репозитория.
В игнорируемом локальном конфиге задайте `mentor.key_file`. Права файла должны
быть 0600. Внутри нужна одна строка: `DEEPSEEK_API_KEY=` и далее ваш ключ.
Если одновременно задана переменная окружения, она имеет приоритет.
Пустой `.env.example` показывает формат и автоматически не загружается.

```bash
python coach.py preflight --config configs/training.local.json --api-smoke
```

Команда отправит setup-запрос, который может учитываться в расходе API.
Наставник получает сводки метрик и параметры управления. Контроллер не прикладывает
тексты промптов датасета и файлы признаков. Для языка объяснений используйте
`mentor.language`: `en` или `ru`.

Решение содержит ровно три поля: `action`, `reason`, `learning_rate`.
Наставник не может исполнять shell-команды. Положительный learning_rate заменяет
базовый темп со следующей порции. У входной проекции сохраняется множитель x3.

## 10. Запуск и остановка

Проверьте свободные GPU и порт сравнения. Если production_service не задан,
контроллер не останавливает чужие inference-процессы.

В текущем терминале:

```bash
python coach.py run --config configs/training.local.json --start-authorized
```

В Linux user-сервисе systemd:

```bash
python coach.py launch --config configs/training.local.json --start-authorized
journalctl --user -u model-improvement-lab.service -f
```

В командах используется имя по умолчанию `model-improvement-lab.service`. Для своего
сервиса задайте `unit_name` в конфигурации и указывайте это имя при просмотре
лога и остановке обучения.

Для user-сервиса задайте закрытый файл ключа. Фоновый systemd-процесс не получает
автоматически все переменные терминала. Launcher добавляет ExecStopPost для
возврата inference. Запуск переживает закрытие SSH, но не является настройкой
возобновления после перезагрузки. Чтобы user-сервис жил после выхода пользователя,
сама машина должна разрешать его systemd manager оставаться активным.

```bash
python coach.py status --config configs/training.local.json
systemctl --user stop model-improvement-lab.service
```

Для foreground используется Ctrl-C. Сохраняется последний принятый checkpoint.
Повторный запуск с тем же конфигом продолжает незавершённую работу.

## 11. Рабочий inference-сервис

Оставьте production_service равным null, если управление сервисом не нужно.
Если оно нужно, задайте имя, scope, health URL и, при необходимости, пути к
config и launcher. Контроллер проверяет активность сервиса и свободные slots,
останавливает его на время GPU-работы, а затем возвращает.

`production_scope: "user"` использует `systemctl --user`. Значение `"system"`
использует системный systemctl и существующее noninteractive sudo-право обычного
пользователя. Пакет не устанавливает sudo rules. Проверьте восстановление своего
сервиса до длительного запуска без присмотра.

## 12. Checkpoint и результаты

После порции сохраняются outputs/last-state.pt и outputs/training.json.
Checkpoint содержит текущие и лучшие веса, optimizer, scaler, RNG, шаг и hash
плана. В экспортированный GGUF записываются лучшие сохранённые веса.

Признаки принятых train-порций удаляются после решения наставника.
Validation-признаки остаются. Незавершённая порция может быть повторена;
принятая не обучается второй раз при возобновлении. Слишком короткие ответы
target исключаются и учитываются в отчёте.

Для нового этапа с переносом optimizer задайте seed_checkpoint,
seed_plan_sha256 и seed_step в новом локальном конфиге с новым run_dir.
Последний шаг seed должен одновременно быть лучшим. Текущие веса должны
совпадать с сохранёнными лучшими. Это не позволяет соединить лучшие веса
с optimizer от другого шага. Загружайте только доверенные PyTorch-checkpoints.

Если есть только лучший GGUF, укажите его как initial_draft и не задавайте seed.
Это старт от готовых весов с новым optimizer.

После остановки обучения контроллер сравнивает no-draft, parent и candidate
на отложенных test-промптах. Выполняются два прохода, до 512 новых токенов на
запрос. Сохраняются hashes ответов, timings и разрез по языкам.
Это не проверка всего окна контекста и не независимая оценка всех решений.
Кандидат автоматически не устанавливается.

## 13. Если что-то пошло не так

| Симптом | Что проверить |
| --- | --- |
| Несовпадение токенов | Template и точный пересчёт до подготовки нового запуска |
| Несовпадение hash | Вернуть зафиксированные файлы либо создать новый эксперимент |
| Неправильный размер тензора | Совместимость архитектур target и draft |
| CUDA out of memory | Размещение target, batch, context и другие GPU-процессы |
| AMP retry | Та же партия повторяется с меньшим loss scale |
| Неконечный forward loss | Данные и веса; обучение останавливается |
| Нет ключа или отказ авторизации | Источник ключа и доступ аккаунта провайдера |
| Ожидание наставника | evidence/mentor-waiting.json и журнал расхода |
| Мало диска | Очистить ненужные файлы вне эксперимента либо выбрать другой диск |
| Inference не вернулся | Сохранить restore-needed и исправить запуск/health сервиса |

После исправления восстановления выполните:

```bash
python coach.py run --config configs/training.local.json --restore-only
```

Не удаляйте маркер восстановления, пока сервис не станет исправным.

## 14. Подготовка к общему доступу

Запустите тесты и privacy-check. Отдельно проверьте список файлов Git.
Поиск шаблонов не доказывает отсутствие любой возможной личной детали.
Не включайте data, models, runs, локальные конфиги, ключи, признаки и логи.
Отчёты реального запуска могут содержать промпты и ответы. Ссылки на публичные
датасеты в исходниках добавлены намеренно как примеры.

Ветки версий и команды публикации описаны в [releases.md](releases.md).
Локальные ветки и теги сами по себе ничего не публикуют.
