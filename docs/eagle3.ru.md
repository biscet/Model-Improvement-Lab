# Офлайн-обучение EAGLE-3

[English version](eagle3.en.md) | [Model Improvement Lab](../README.ru.md)

> [!IMPORTANT]
> Для офлайн-обучения нужны скрытые состояния целевой языковой модели.
> В готовом рецепте выполните описанный ниже захват перед обучением.
> Для другой поддерживаемой модели захватите признаки самостоятельно.
> Токенизатор, слои и формат признаков должны совпадать с её рецептом SpecForge.
> [Подготовка данных в SpecForge](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/data_preparation.md)

В готовом рецепте llama.cpp сохраняет скрытые состояния [Whittle Q5_K_M GGUF](https://huggingface.co/logic65/Whittle-Qwen-3.8-35B-A3B-GGUF/blob/main/Whittle-Qwen-3.8-35B-A3B-Q5_K_M.gguf).
SpecForge обучает однослойный EAGLE-3 draft на этих признаках, используя
замороженные embedding и LM head из того же GGUF.

## Настройка готового рецепта на Ubuntu

Нужны Python 3.12, CUDA 12.9 и одна NVIDIA Tesla V100 с 32 ГБ видеопамяти.
Если подходящих карт несколько, задайте UUID выбранной карты в
`CUDA_VISIBLE_DEVICES`. Выполняйте команды из корня Model Improvement Lab.
`SF` и `LLAMA` ниже указывают на новые рабочие копии под рецепт Whittle.

```bash
SF="$PWD/runtime/eagle3-specforge"
LLAMA="$PWD/runtime/eagle3-llama"
PATCHES="$PWD/patches/eagle3"
JOB="$PWD/runs/eagle3"

git init "$SF"
git -C "$SF" fetch --depth=1 https://github.com/sgl-project/SpecForge.git \
  91d8a5a5443c5b93a905f8eead79f23a00311a41
git -C "$SF" checkout --detach FETCH_HEAD
git -C "$SF" apply "$PATCHES/specforge-whittle.patch"

git init "$LLAMA"
git -C "$LLAMA" fetch --depth=1 https://github.com/ggml-org/llama.cpp.git \
  52975342ff6b85127ff31930d39e36725fd08809
git -C "$LLAMA" checkout --detach FETCH_HEAD
git -C "$LLAMA" apply "$PATCHES/llama.cpp-whittle-v100.patch"
```

Укажите путь к CUDA Toolkit в `CUDA_HOME`. В примере библиотеки лежат в
`lib64`. Если они установлены в `lib`, поменяйте путь в команде сборки.

```bash
export PATH="$CUDA_HOME/bin:$PATH"
cmake -S "$LLAMA" -B "$LLAMA/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70-real \
  -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_BUILD_RPATH="$CUDA_HOME/lib64" \
  -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  '-DGGML_CUDA_FA_QUANTS=f16-f16;q8_0-q8_0;q4_0-q4_0' \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_UI=OFF
cmake --build "$LLAMA/build" --target llama-server llama-whittle-eagle3-capture -j 6

mkdir -p "$JOB/target"
python3.12 -m venv "$JOB/.venv"
PY="$JOB/.venv/bin/python"
"$PY" -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -r requirements-eagle3.txt "$LLAMA/gguf-py"
"$PY" -c 'import torch; print(torch.__version__, torch.cuda.get_arch_list())'
```

В списке архитектур PyTorch должна быть `sm_70`. Coach использует это
окружение и загружает SpecForge из рабочей копии с применённым патчем.

## Подготовка target и входных токенов

Сохраните конфигурацию и токенизатор Whittle в `$JOB/target`.

```bash
HF_REVISION=67272fd284ef110586a73f646fda855a60a16e69
for file in config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
  curl -fL "https://huggingface.co/whittleOrg/Whittle-Qwen-3.8-35B-A3B/resolve/$HF_REVISION/$file" \
    -o "$JOB/target/$file"
done
```

Укажите в `MODEL` абсолютный путь к Whittle Q5_K_M GGUF, а в `JSONL`
путь к последовательностям, подготовленным токенизатором этой модели.
В каждой строке нужны `input_ids` и бинарная маска `loss_mask` той же длины.
Можно передать `prompt_n` вместо маски, если обучаться нужно на всех позициях
после промпта. Команда `download-example` готовит промпты. Для EAGLE-3
подготовьте полные последовательности токенов.

## Захват и обучение

```bash
python3 coach.py eagle3 target --specforge "$SF" --python "$PY" \
  --work-dir "$JOB" --model "$MODEL"
python3 coach.py eagle3 capture --specforge "$SF" --python "$PY" \
  --work-dir "$JOB" --model "$MODEL" \
  --exporter "$LLAMA/build/bin/llama-whittle-eagle3-capture" --input "$JSONL"
python3 coach.py eagle3 train --specforge "$SF" --python "$PY" --work-dir "$JOB"
```

Повторный захват с теми же входами и настройками пропустит готовые записи.
Для проверки одного шага захватите хотя бы четыре пригодных записи и добавьте
`--max-steps 1` к `train`. После загрузки конфигурации target команда
`train --plan` с теми же путями проверяет разрешение рецепта на CPU.

Для возобновления передайте `--resume "$CHECKPOINT"` в `train` и используйте
тот же кеш признаков и рецепт. Держите рабочие каталоги, веса, признаки и
checkpoint вне отслеживаемых исходников.

## Экспорт в GGUF

Укажите checkpoint SpecForge в `CHECKPOINT` и новый путь с расширением
`.gguf` в `DRAFT`.

```bash
python3 coach.py eagle3 export --specforge "$SF" --python "$PY" \
  --work-dir "$JOB" --runtime "$LLAMA" --checkpoint "$CHECKPOINT" --output "$DRAFT"
```

Для инференса используйте пропатченный `llama-server` с параметрами
`--spec-type draft-eagle3 --spec-draft-model "$DRAFT"`. Захват и инференс
используют один формат признаков Whittle, `native-attn-mix-v1`.

В ML-Docs описаны формат признаков, сдвиг токенов и исходный короткий прогон
на V100. [Офлайн-обучение](https://github.com/biscet/ML-Docs/blob/main/docs/eagle3/offline-v100.ru.md)
· [Признаки Whittle](https://github.com/biscet/ML-Docs/blob/main/docs/eagle3/whittle-features.ru.md)
