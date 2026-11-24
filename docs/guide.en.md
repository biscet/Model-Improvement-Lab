# Set up and use Model Improvement Lab 3.0.0

[Русская версия](guide.ru.md) | [Project overview](../README.md)

This guide covers the bundled DFlash2 recipe for a local language model.
The controller captures target features as part of `coach.py run`.

## 1. Choose where to run each part

Use macOS or Linux for the offline demo, source checks and reading run status.
Real capture and training need Linux, an NVIDIA CUDA setup and compatible model
files. The Mac copy of this repository is not a running GPU training service.

The example configuration uses two visible GPUs. Target compute and the draft
use the first one. The target's per-layer token embedding table uses the second
one. Set `gpu_uuids` to your device indices or GPU UUIDs. If you use one GPU,
also change the benchmark arguments in `server_argv` to match that layout.

The bundled DFlash2 recipe uses these values:

| Item | Value |
| --- | --- |
| Target feature layers, one-based | 1, 6, 11, 16, 22, 27, 32, 37 |
| Feature width per layer | 2048 |
| Target vocabulary | 248320 |
| Draft transformer layers | 6 |
| Training precision | FP32 master weights, FP16 autocast |
| Feature files | FP16, 8 x 2048 values per captured token |

The bundled recipe expects a target compatible with Qwen3.8 35B A3B.
For a different target architecture, provide a matching tokenizer, feature
capture, draft model, and training recipe.

## 2. Check the source copy without installing anything

From the repository root:

```bash
python3 coach.py demo --run-dir ../model-improvement-lab-demo
python3 coach.py status --run-dir ../model-improvement-lab-demo
python3 -m unittest discover -s tests -v
python3 coach.py privacy-check
```

The demo is a simulation. It creates 24 synthetic inputs, three shards and
synthetic review decisions. It performs no training and sends no requests.
Some tests are skipped if PyTorch is absent. This is expected on a plain Mac.

## 3. Install the Linux training environment

Use Python 3.11 or 3.12 for the training environment. Keep the source folder on
a filesystem with enough space for target files, captured features and atomic
checkpoint replacement. Space for one feature shard is not the total disk cost.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The appropriate PyTorch CUDA wheel depends on your driver and GPU. Check that
the installed build sees CUDA before proceeding:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
nvidia-smi
```

The example requirements record the training library versions. They do not
install a GPU driver, CUDA toolkit or model files.

## 4. Build the compatible runtime and capture tools

The capture code uses extension APIs from
[mistrjirka/llama.cpp](https://github.com/mistrjirka/llama.cpp). Stock llama.cpp
may not provide those APIs. The reference source revision is
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

The patch averages the target's hyper-connection streams for feature capture.
Do not change that feature definition during a run. A different runtime may
need an adapted patch. Treat a build error or a failed smoke test as a setup
problem, not as a reason to omit the patch.

Set CUDA architecture options when your toolkit needs them. For a nonstandard
CUDA installation, set the required library paths through the configuration's
`environment` object. Do not copy another computer's library paths blindly.

## 5. Provide compatible model files and a local configuration

```bash
mkdir -p models
cp configs/example.json configs/training.local.json
```

Place your compatible target at `models/target.gguf` and your initial F16 draft
at `models/draft.gguf`, or change the two paths. The trainer reads the frozen
target embedding and output head from GGUF. It does not require a separate
full-precision target checkpoint.

Every relative path in JSON is relative to the JSON file. For example,
`../models/target.gguf` in `configs/training.local.json` refers to the repository's
`models` folder. Keep the `.local.json` suffix so Git ignores your configuration.

Review these settings before using the example:

| Setting | Meaning |
| --- | --- |
| `run_dir` | A new folder for this run, or an existing folder when resuming |
| `dataset` | Folder containing train.jsonl, validation.jsonl and test.jsonl |
| `python` | Python executable from the training virtual environment |
| `capture_binary`, `tokenize_binary`, `server_binary` | Built executables |
| `gpu_uuids` | Visible devices in the intended CUDA order |
| `capture_context` | Must fit the largest prompt plus generated tokens |
| `training_objective` | `uniform` (default) or `dpace` |
| `shard_feature_budget_gib` | Upper bound for a train shard's feature files |
| `validation_monitor_rows`, `benchmark_test_rows` | Reserved sample counts |
| `max_epochs` | Maximum full passes; null lets the mentor request more passes |
| `server_argv` | Complete argument list for benchmark inference |
| `production_service` | Optional inference service to stop and restore |

The example allows at most one pass. The mentor can stop after an earlier shard.
There is no configured total API-token budget. Transient API errors are retried;
invalid authorization ends the run. Review provider usage and costs yourself.

For D-PACE, set `"training_objective": "dpace"` before running `prepare`.
The recipe smooths each gold-token probability with α = 0.5, multiplies
confidences along the prefix and weights each position by the sum of
confidences for prefixes that include it. The weights are detached from
autograd. They apply to token NLL and to selector NLL where the draft
top-16 contains the gold token. The weighted sum is averaged over blocks.

Validation and checkpoint selection use the uniform loss so that runs
share a common scale. The training objective is frozen with the run plan.
To switch objectives, prepare a new `run_dir`. See the paired experiment
and its loss formula. [D-PACE in ML-Docs](https://github.com/biscet/ML-Docs/blob/main/docs/experiments/position-weighting.md)

## 6. Choose a public dataset example

The repository contains references to these public datasets:

- [DFlash DeepSeek TraceMix](https://huggingface.co/datasets/biscet/DFlash-DeepSeek-TraceMix).
- [DFlash ChatGPT TraceMix Lite](https://huggingface.co/datasets/biscet/DFlash-ChatGPT-TraceMix-Lite).

Pinned revisions and file paths are in `examples/huggingface-datasets.json`.
The downloader does not require a Hugging Face token for these public files.
It stores normalized prompt contexts, not the supplied completions or raw
provenance payloads. It does not upload anything.

Choose one command and use an empty destination:

```bash
python coach.py download-example --example deepseek --output data/example
```

For only the separate 272-context training supplement:

```bash
python coach.py download-example --example stage2 --output data/example
```

For the smaller OpenMix example:

```bash
cp configs/lite.example.json configs/training.local.json
python coach.py download-example --example lite --output data/example
```

The Lite source has train and validation files. The downloader reserves whole
validation source families for a new test split. It does not move source train
records into that test split. The smaller config uses 16 validation and 16 test
records. The Stage-2 example uses the main dataset's validation and test files.

These splits do not prove independence from every model's earlier training.
Check overlap with your parent draft's training history when that history is
available. Source notices and third-party terms still apply to the datasets.

## 7. Use your own inputs

Provide UTF-8 `train.jsonl`, `validation.jsonl` and `test.jsonl` in the configured
dataset folder. One record is one rendered target-model prompt. A synthetic
record showing the fields is in `examples/input-record.json`.

Required fields are `id`, `project`, `source_path`, `source_sha256`,
`sampling_group`, `prompt` and `language`. Token counts are added in the next
step. IDs may contain letters, digits, underscores and hyphens. Related source
files and language variants must remain in the same split. Use a stable source
hash and source path so the checks can detect known overlap.

Render prompts with the target's own chat template before tokenization. The
public examples use their recorded template. Retokenizing cannot fix a wrong
chat template or make labels from a different teacher compatible with this
target. Captured labels are always regenerated by the configured target.

## 8. Count tokens and freeze the plan

```bash
python coach.py tokenize-inputs --config configs/training.local.json
python coach.py prepare --config configs/training.local.json
python coach.py verify-tokens --config configs/training.local.json
```

`tokenize-inputs` updates exact prompt lengths on CPU. This is required for Lite,
which has no stored prompt token counts. `prepare` checks split boundaries and
creates the shard plan. It hashes the inputs, binaries, code and model files.
`verify-tokens` checks the frozen capture inputs again. These steps do not start
GPU capture, training or DeepSeek calls.

After preparation, do not edit the inputs, code, runtime, model files or config
for that run. A changed hash causes preflight to stop. Use a new run folder for
a changed experiment. Do not bypass the check by editing `plan.json`.

## 9. Configure DeepSeek

The example uses `deepseek-flash` at the DeepSeek API. Your account must support
the alias and the request fields. The model name is configurable.

For a foreground session, set `DEEPSEEK_API_KEY` in the process environment.
Do not put its value in a committed JSON file, command argument or README.

For a background user service, use a private file outside the repository. Set
`mentor.key_file` in your ignored local config to that file's path. The file
must have mode 0600 and contain one line with `DEEPSEEK_API_KEY=` followed by
your key. The environment variable takes precedence when both are present.
The empty `.env.example` is a format example and is not loaded automatically.

```bash
python coach.py preflight --config configs/training.local.json --api-smoke
```

This command sends a setup request and can incur API usage. The mentor receives
validation summaries and control settings. Dataset prompt text and target
feature files are not attached by the controller. Use `mentor.language` with
`en` or `ru` for the explanations.

The mentor returns exactly three fields: `action`, `reason` and `learning_rate`.
It cannot execute shell commands. A positive learning rate replaces the base
rate for the next shard. The projection group keeps its x3 multiplier.

## 10. Start and stop

Check that the chosen GPUs and benchmark port are free. With no production
service configured, the controller does not stop another inference process.

Foreground launch:

```bash
python coach.py run --config configs/training.local.json --start-authorized
```

Linux systemd user-service launch:

```bash
python coach.py launch --config configs/training.local.json --start-authorized
journalctl --user -u model-improvement-lab.service -f
```

The commands use the default unit `model-improvement-lab.service`. For a deployment with
a custom unit, set `unit_name` in the configuration and use that name when
viewing logs or stopping training.

Use a private mentor key file for the user-service launch. User services do not
automatically inherit all variables from the launching terminal. The launcher
adds an `ExecStopPost` restoration hook. It survives SSH disconnection; it is
not a reboot-resume service. To keep a user service alive after logout, the
machine must already allow that user's systemd manager to remain active.

```bash
python coach.py status --config configs/training.local.json
systemctl --user stop model-improvement-lab.service
```

Ctrl-C stops a foreground run. The controller retains the last committed
checkpoint. The next launch with the same command resumes an unfinished run.

## 11. Manage an existing inference service

Leave `production_service` null unless you want the controller to manage one.
If enabled, set the service name, scope, health URL and optional config/launcher
paths. The controller checks that the service is active and no slot is busy.
It then stops the service for GPU work and restores it on exit.

`production_scope: "user"` uses `systemctl --user`. `"system"` uses system
systemctl and, for a normal user, the existing noninteractive sudo permission.
This repository does not install sudo rules. Test restoration on your setup
before leaving a long run unattended.

## 12. Understand checkpoints and results

Each completed shard saves `outputs/last-state.pt` and `outputs/training.json`.
The checkpoint includes current weights, best weights, optimizer, scaler, RNG,
step and the frozen plan hash. The exported GGUF contains the best saved weights.

Feature files from accepted train shards are deleted after review. Validation
features stay available. An incomplete shard can be replayed; an already
committed shard is not trained a second time on resume. Too-short target
generations are excluded and counted in the report.

To start a new stage with optimizer recovery, set `seed_checkpoint`,
`seed_plan_sha256` and `seed_step` in a new local config with a new `run_dir`.
The seed's last step must also be its best step, and its current weights must
match its saved best weights. This prevents pairing best weights with an
optimizer from another step. Only load trusted PyTorch checkpoints.

If only a best GGUF is available, use it as `initial_draft` and omit the seed
fields. That is a weight warm start; it starts a new optimizer.

After training stops, the controller compares no-draft, parent and candidate
on held-out test prompts. It uses two passes and up to 512 new tokens per
request. Results include response hashes, timings and language slices.
It does not validate the target's full context window or independently score
all generated solutions. It does not install the candidate.

## 13. Find a failure

| Symptom | Where to look or what to change |
| --- | --- |
| Prompt token mismatch | Rerender or recount before preparing a new run |
| Hash mismatch | Restore the frozen files or create a new experiment |
| Tensor shape mismatch | Check target and draft architecture compatibility |
| CUDA out of memory | Review target placement, batch, context and other GPU processes |
| AMP retry | The same batch is retried with a smaller loss scale |
| Non-finite forward loss | Inspect the data and model; the run stops |
| Missing API key or authorization error | Check the private key source and provider account |
| Waiting for mentor | See evidence/mentor-waiting.json and the usage ledger |
| Disk capacity failure | Free disposable files outside the experiment or choose another run disk |
| Inference not restored | Keep evidence/restore-needed and fix service startup/health |

Run `python coach.py run --config configs/training.local.json --restore-only`
after fixing restoration settings. Do not delete the restoration marker before
the target service is healthy.

## 14. Prepare a public release

Run the tests and `privacy-check`. Review the actual Git file list as well.
The scanner reports common patterns, not every possible identifying detail.
Keep `data`, `models`, `runs`, local configs, keys, feature files and logs out of
the source release. New runtime reports may contain prompts and responses.
The public dataset links in this source tree are deliberate examples.

Version branches and publication commands are in [releases.md](releases.md).
Creating a local branch or tag does not publish the repository.
