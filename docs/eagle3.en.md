# Offline EAGLE-3 training

[Русская версия](eagle3.ru.md) | [Model Improvement Lab](../README.md)

> [!IMPORTANT]
> Offline training requires hidden states from the target language model.
> Run the capture command below before training with the bundled recipe.
> For another supported target, capture its features separately using the
> tokenizer, layers, and format required by its SpecForge recipe.
> [SpecForge data preparation](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/data_preparation.md)

In the bundled recipe, llama.cpp captures the hidden states of the [Whittle Q5_K_M GGUF](https://huggingface.co/logic65/Whittle-Qwen-3.8-35B-A3B-GGUF/blob/main/Whittle-Qwen-3.8-35B-A3B-Q5_K_M.gguf).
SpecForge trains a one-layer EAGLE-3 draft from those features using the frozen
embedding and output head extracted from the same GGUF file.

## Set up the bundled recipe on Ubuntu

Use Python 3.12, CUDA 12.9, and one NVIDIA Tesla V100 with 32 GB of VRAM. Select
the GPU by UUID with `CUDA_VISIBLE_DEVICES` if multiple cards qualify. Run the
commands from the Model Improvement Lab root. `SF` and `LLAMA` below are new checkouts
dedicated to the Whittle recipe.

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

Set `CUDA_HOME` to the installed toolkit. The build below assumes its libraries
are under `lib64`. Use `lib` if that is where they are installed.

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

The PyTorch build must list `sm_70`. Coach loads SpecForge from the patched
checkout and uses this virtual environment for its EAGLE-3 stages.

## Prepare the target and tokenized inputs

Place the configuration and tokenizer for the Whittle model in `$JOB/target`.

```bash
HF_REVISION=67272fd284ef110586a73f646fda855a60a16e69
for file in config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
  curl -fL "https://huggingface.co/whittleOrg/Whittle-Qwen-3.8-35B-A3B/resolve/$HF_REVISION/$file" \
    -o "$JOB/target/$file"
done
```

Set `MODEL` to the absolute path of your Whittle Q5_K_M GGUF. Set `JSONL` to
an absolute path containing sequences tokenized with the matching tokenizer.
Each record has `input_ids` plus a binary `loss_mask` of the same length.
Use `prompt_n` instead of the mask to supervise all positions after the prompt.
The public dataset downloader provides prompt contexts. Prepare completed token
sequences before using them for EAGLE-3.

## Capture and train

```bash
python3 coach.py eagle3 target --specforge "$SF" --python "$PY" \
  --work-dir "$JOB" --model "$MODEL"
python3 coach.py eagle3 capture --specforge "$SF" --python "$PY" \
  --work-dir "$JOB" --model "$MODEL" \
  --exporter "$LLAMA/build/bin/llama-whittle-eagle3-capture" --input "$JSONL"
python3 coach.py eagle3 train --specforge "$SF" --python "$PY" --work-dir "$JOB"
```

Repeat capture with the same inputs and settings to skip completed records.
For a one-step training check, capture at least four usable records and add
`--max-steps 1` to `train`. Run `train --plan` with the same paths after
placing the target configuration to resolve the recipe on CPU.

To resume, pass `--resume "$CHECKPOINT"` to `train` and use the same feature
cache and recipe. Keep job directories, weights, features and checkpoints
outside the tracked source tree.

## Export to GGUF

Set `CHECKPOINT` to a SpecForge checkpoint and `DRAFT` to a new `.gguf` path.

```bash
python3 coach.py eagle3 export --specforge "$SF" --python "$PY" \
  --work-dir "$JOB" --runtime "$LLAMA" --checkpoint "$CHECKPOINT" --output "$DRAFT"
```

Serve the draft with the patched `llama-server` using
`--spec-type draft-eagle3 --spec-draft-model "$DRAFT"`. Capture and inference
use the same Whittle feature layout, `native-attn-mix-v1`.

ML-Docs covers the feature layout, token alignment and original V100 smoke run.
[Offline training](https://github.com/biscet/ML-Docs/blob/main/docs/eagle3/offline-v100.md)
· [Whittle features](https://github.com/biscet/ML-Docs/blob/main/docs/eagle3/whittle-features.md)
