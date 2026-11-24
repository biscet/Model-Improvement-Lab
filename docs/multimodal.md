# Multimodal experiments

Model Improvement Lab supports companion-model workflows through `coach.py multimodal`.
Use a dedicated Python environment with `requirements-multimodal.txt`.
The recipes use pinned upstream checkpoints and local media files.

| Recipe | Task | Training method | Dataset fields |
| --- | --- | --- | --- |
| `configs/vision.example.json` | Image understanding | LoRA on attention query/value projections | `image`, `prompt`, `text` |
| `configs/speech.example.json` | Speech transcription | LoRA on attention query/value projections | `audio`, `language`, `text` |
| `configs/audio.example.json` | Audio/text embeddings | Contrastive training of both projection heads and temperature | `audio`, `text` |
| `configs/music.example.json` | Free-form music descriptions | Music Flamingo LoRA on attention query/value projections | `audio`, `prompt`, `text` |

Copy a recipe to a `.local.json` file. Paths resolve relative to that file.
Set `model_dir`, `dataset`, `run_dir`, and the intended publisher/model name.
Training and inference use Transformers checkpoints. The Qwen3-ASR recipe uses
Qwen's native Transformers conversion, with the `-hf` suffix.

```bash
python coach.py multimodal download --config configs/vision.local.json
python coach.py multimodal validate --config configs/vision.local.json
python coach.py multimodal infer --config configs/vision.local.json --input photo.png --prompt "Describe the image."
python coach.py multimodal train --config configs/vision.local.json
python coach.py multimodal train --config configs/vision.local.json --resume
python coach.py multimodal status --config configs/vision.local.json
```

Select a training GPU with `CUDA_VISIBLE_DEVICES` before launching Python and
set the recipe's device to `cuda:0`. Prefer the device UUID reported by
`nvidia-smi -L`; numeric CUDA ordinals can differ from the displayed GPU order.
FP16 supports the V100. CLAP uses FP32.
Launch experiments sequentially when sharing GPU memory.

Each JSONL line is one example. Media paths resolve relative to the JSONL file.

```json
{"image":"images/diagram.png","prompt":"Explain the diagram.","text":"The diagram shows..."}
{"audio":"audio/recording.wav","language":"Russian","text":"Точная расшифровка записи."}
{"audio":"audio/piano.wav","text":"A piano plays a quiet melody."}
{"audio":"audio/song.wav","prompt":"Describe the instrumentation, genre and mood.","text":"A restrained instrumental arrangement combines..."}
```

Use the matching record format for each task. Generative training masks prompt,
media and padding tokens from the loss. It supports one example per microbatch
and configurable gradient accumulation. Oversized examples raise a clear error.
CLAP uses batches of at least two distinct audio/caption pairs with a symmetric
contrastive loss. Training clips are at most 10 seconds. Split longer recordings
and give each clip an accurate caption. Inference handles longer audio as
10-second segments and returns their normalized mean embedding.

```bash
python coach.py multimodal infer --config configs/audio.local.json --input clip.wav --labels "A person speaking" "Music playing" --output audio.json
python coach.py multimodal infer --config configs/speech.local.json --input speech.wav --output transcript.json
python coach.py multimodal infer --config configs/music.local.json --input song.wav --output description.json
```

CLAP reports cosine similarities. These scores measure similarity to the
provided descriptions. Qwen-ASR returns a transcript. Vision and Music Flamingo
return generated text. Music Flamingo's primary language is English. It processes
audio in 30-second windows, up to its 20-minute model limit. Training uses the
recipe's token limit and short clips are suitable for initial experiments.
Supply `--adapter RUN/candidate` to test trained weights or a LoRA adapter.

The default cumulative training budget is 10,800 seconds. Model Improvement Lab checks it
after each optimizer step and requests a checkpoint before stopping. A single
step and final checkpoint writing can extend past the deadline. Checkpoints
include optimizer, scaler, random state and Trainer progress. Resume verifies
the pinned model, recipe and hashes of both JSONL records and media. Save
frequency is configurable with `save_steps`; abrupt failures can lose work
after the last complete checkpoint. A file lock prevents concurrent writers.

Optionally set `validation_dataset` to a separate JSONL file with the same
record format. Training rejects overlapping media hashes and records the
validation fingerprint for resume. Set matching `eval_steps` and `save_steps`
to evaluate loss and save checkpoints periodically. The exported candidate then
uses the checkpoint with the lowest validation loss; this is checkpoint
selection, not a comparison with the upstream model. `warmup_steps` optionally
ramps the learning rate before holding it constant. These evaluations count
toward the cumulative wall-clock budget.

For a minimal installation check, use `max_steps=1`,
`gradient_accumulation_steps=1`, `save_steps=1` and a separate verification run.
Training reports successful and skipped optimizer updates separately. A run
with no successful update, or nonfinite trained parameters, fails without
exporting a candidate. FP16 starts with a conservative loss scale; resumed runs
restore their saved scaler state. This checks an optimizer update and checkpoint
writing. An inference call with
the resulting candidate checks reloading. Keep verification candidates separate
from models intended for release.

Stage a local HF package after choosing the desired base or trained candidate.

```bash
python coach.py multimodal package --config configs/audio.local.json --output ../exports/audio-base
python coach.py multimodal package --config configs/vision.local.json --trained --output ../exports/vision-adapter
```

The package includes weights or adapters, processor files, source revision,
upstream documentation, the original license, attribution, training provenance
when applicable, and checksums. Base packages are marked as upstream weights.
LoRA metadata points to the original HF model so it can be loaded on another
machine. Publication is a separate operation using the selected HF account.

Music Flamingo uses NVIDIA's OneWay Noncommercial License. Its derivatives retain
these terms, including the limitation to academic, non-commercial use. The recipe records that license separately and the package includes
the upstream license documents. The Qwen3.5, Qwen3-ASR and CLAP recipes use Apache-2.0.

Choose a fine-tuning dataset when representative examples show a task-specific
gap. Assess improvement on held-out examples before describing a release as
better than its source checkpoint.
