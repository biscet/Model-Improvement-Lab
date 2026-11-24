# Model Improvement Lab

Version 3.0.0. [Russian instructions](README.ru.md).

> [!NOTE]
> **Experimental community release.** This release is intended for research
> and hands-on experiments with model fine-tuning and speculative decoding.

Model Improvement Lab is a local research toolkit for running, fine-tuning and
packaging vision, speech, music and draft models. It includes pretrained-model
inference, bounded fine-tuning and Hugging Face package preparation, alongside
DFlash2 with D-PACE, offline EAGLE-3 and GGUF feature capture.

GPU training requires a compatible model, capture runtime, and recipe. The
bundled recipes run on Linux with CUDA. The offline demo, status command and
most tests also work on macOS.

## What is included

- Vision, speech transcription, free-form music descriptions and audio/text embeddings.
- Multimodal fine-tuning with resume and provenance-aware Hugging Face packaging.
- Capture and token-count tools for a compatible llama.cpp fork.
- A full draft trainer with checkpoint, optimizer, scaler and RNG recovery.
- Uniform and D-PACE training objectives for DFlash2.
- Offline EAGLE-3 capture, training and GGUF export with a V100 recipe.
- A controller that processes feature files in bounded shards.
- DeepSeek review, decision history and learning-rate changes.
- A benchmark for no-draft, parent draft and candidate draft.
- Tests, synthetic demo inputs and public dataset examples.

No model weights, private datasets, training logs or credentials are included.
The browser dashboard is not implemented in this version.

## Try the offline demo

Python 3.10 or newer is enough. No packages or API key are needed.

```bash
python3 coach.py demo --run-dir ../model-improvement-lab-demo
python3 coach.py status --run-dir ../model-improvement-lab-demo
python3 -m unittest discover -s tests -v
python3 coach.py privacy-check
```

The demo writes synthetic files outside the source folder. It does not train a
model or call DeepSeek. Tests that need PyTorch are skipped when it is absent.
Choose a new demo directory for another run.

## Train a DFlash2 draft

Follow the [English guide](docs/guide.en.md). It covers the runtime build, CUDA
environment, data format, API key, preparation, launch, resume and evaluation.
The [Russian guide](docs/guide.ru.md) covers the same steps.

Start with `configs/example.json`. Copy it to `configs/training.local.json`
before changing paths. Paths in JSON are resolved relative to that JSON file.
The small dataset example uses `configs/lite.example.json` instead.
`coach.py run` drives feature capture as part of the bundled DFlash2 workflow.
For a different target architecture, provide a compatible capture tool and
DFlash2 recipe.

Set `"training_objective": "dpace"` before preparing a new run to use
D-PACE. The default is `uniform`. The DFlash2 guide explains how D-PACE
weights token and selector losses.

## Train an EAGLE-3 draft

`python3 coach.py eagle3 --help` lists the capture, training and export stages.
The [EAGLE-3 guide](docs/eagle3.en.md) shows how to run the bundled Whittle
recipe on a V100 with SpecForge and llama.cpp.

> [!IMPORTANT]
> EAGLE-3 training reads precomputed target features. Run `eagle3 capture`
> before `eagle3 train` for the bundled recipe. For another supported target,
> capture its features separately and use a matching
> [SpecForge recipe](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md).

ML-Docs covers the D-PACE experiment and EAGLE-3 feature layout.
[ML-Docs](https://github.com/biscet/ML-Docs)

## Vision, speech and audio experiments

`python3 coach.py multimodal --help` provides download, dataset validation,
inference, bounded fine-tuning, resume and Hugging Face package staging.
The initial recipes cover Qwen3.5-4B vision LoRA, Qwen3-ASR-0.6B speech LoRA,
Music Flamingo music-caption LoRA, and CLAP audio/text projection training.
See [the multimodal guide](docs/multimodal.md)
for formats and commands.

## Public dataset examples

- [DFlash DeepSeek TraceMix](https://huggingface.co/datasets/biscet/DFlash-DeepSeek-TraceMix).
  Use `deepseek` for the main task splits or `stage2` for the separate 272-context
  training supplement with the main validation and test splits.
- [DFlash ChatGPT TraceMix Lite](https://huggingface.co/datasets/biscet/DFlash-ChatGPT-TraceMix-Lite).
  The `lite` example uses OpenMix. It reserves part of the source validation
  families for testing because that subset has no separate test file.

Only links, pinned revisions and download settings are stored here. Dataset
records are downloaded only when the user runs `download-example`. Existing
completions are discarded. Labels and features for training come from the
configured target model.

## Limits of this version

The portable controller has CPU and mocked integration tests. Its training
backend comes from an existing CUDA implementation. A complete GPU run of this
portable package is still needed before treating a new setup as verified.

The recipe fixes the feature layers, vocabulary and draft architecture in code.
Changing `target_model` alone does not make another architecture compatible.
The example stops after at most one full pass; DeepSeek can stop earlier. Loss
and prefix metrics are offline proxies. They do not establish serving speed or
response quality. The candidate is not installed into an inference service.

## Versions and licenses

The architecture is documented in the
[Russian architecture guide](docs/architecture.ru.md).

`main` contains v3.0.0. The `v2.0.1`, `v2.0.0` and `v1.0.0` branches preserve earlier releases.
See the [changelog](CHANGELOG.md) and [release steps](docs/releases.md).

Original utilities use Apache-2.0. The upstream DFlash model implementation uses
MIT. See [NOTICE](NOTICE), [LICENSE](LICENSE) and [DFlash MIT](licenses/DFlash-MIT.txt).
