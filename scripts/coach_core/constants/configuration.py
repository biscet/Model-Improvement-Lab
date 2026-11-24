"""Configuration defaults and filesystem settings."""

from scripts.coach_core.constants.paths import PROJECT_ROOT
from scripts.coach_core.constants.recipe import UNIFORM_OBJECTIVE
from scripts.coach_core.types import Config

DEFAULT_MODEL_NAME = "ModelImprovementLab-candidate"
DEFAULT_UNIT_NAME = "model-improvement-lab"

DEFAULT_CONFIG: Config = {
    "base_dir": str(PROJECT_ROOT),
    "python": "python3",
    "gpu_uuids": ["0"],
    "production_service": None,
    "production_config": None,
    "production_launcher": None,
    "production_scope": "user",
    "production_url": "http://127.0.0.1:1234",
    "benchmark_url": "http://127.0.0.1:1238",
    "initial_epoch": 0,
    "max_shard_rows": 256,
    "shard_feature_budget_gib": 8,
    "validation_monitor_rows": 256,
    "benchmark_test_rows": 128,
    "max_new_tokens": 320,
    "capture_context": 14336,
    "training_context": 384,
    "training_objective": UNIFORM_OBJECTIVE,
    "batch_size": 4,
    "learning_rate": 2.5e-6,
    "output_name": "draft-candidate-F16.gguf",
    "model_name": DEFAULT_MODEL_NAME,
}
PATH_FIELDS = (
    "base_dir",
    "run_dir",
    "dataset",
    "target_model",
    "initial_draft",
    "capture_binary",
    "tokenize_binary",
    "server_binary",
    "production_config",
    "production_launcher",
    "seed_checkpoint",
    "gguf_python_dir",
)
