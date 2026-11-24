# Changelog

## Unreleased

- Added optional disjoint validation data, validation-loss checkpoint selection and learning-rate warmup for multimodal training.
- Count successful optimizer updates separately from AMP-skipped steps, and reject candidates with no updates or nonfinite parameters.
- Use a conservative initial FP16 loss scale and document GPU selection by UUID.

## 3.0.0

- Renamed the project to Model Improvement Lab and the repository to ModelImprovementLab.
- Updated English and Russian documentation, CLI branding, example names and service defaults.
- Added pretrained inference, dataset validation and bounded fine-tuning for vision, speech, music descriptions and audio/text embeddings.
- Added checkpoint continuation and Hugging Face package staging with source revisions, licenses and provenance.
- Expanded publication exclusions for local configurations, datasets, media, logs and model artifacts.

## 2.0.1

- Renamed the project to DraftLab and updated its CLI and default names.
- Clarified feature capture requirements for local models and the bundled training recipes.

## 2.0.0

- Added D-PACE as an optional DFlash2 training objective.
- Added offline EAGLE-3 capture, training and export, with a Whittle/V100 recipe and patches for SpecForge and llama.cpp.
- Updated the Lite dataset profile and guides to DFlash ChatGPT TraceMix Lite.
- Grouped CLI commands and application code by domain around the coach.py entry point.
- Split preparation, preflight, mentor reviews, training and benchmark into focused stages.
- Separated native capture orchestration, feature extraction and metadata I/O.
- Separated portable configuration, storage, planning, metrics and mentor evidence.
- Split training sample loading, objective computation, model I/O and weight snapshots.
- Extracted DFlash layers and sampling while preserving model state keys and entry points.
- Centralized recipe constants and unified Python/C++ formatting.
- Included nested Python modules in frozen plans.

## 1.0.0

- Added a portable configuration for capture, training and comparison.
- Added DeepSeek validation review and learning-rate decisions.
- Added shard planning, checkpoint continuation and inference service restoration.
- Added a standalone CUDA trainer and GGUF capture sources.
- Added CPU tests, an offline demo and a publication content check.
- Added English and Russian instructions.
- Added pinned examples for two public Hugging Face datasets.

This release contains source code and examples. It contains no private run
results, datasets, model files or credentials. Full GPU acceptance of the
portable package remains to be done.
