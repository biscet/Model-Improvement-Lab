import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.commands import run as supervisor


class CoordinatorTests(unittest.TestCase):
    def exercise(self, decisions, resumed=None, strategy=None, limit=None):
        import hashlib
        from types import SimpleNamespace

        from scripts.coach_core.files import atomic_json, write_rows

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            run = base / "run"
            for name in ("data", "evidence", "outputs", "features", "shards", "configs"):
                (run / name).mkdir(parents=True)
            (base / "evidence").mkdir()
            prod = base / "production.json"
            prod.write_text("{}")
            train = {"id": "t", "prompt_tokens": 10, "split": "train", "dataset_split": "train"}
            val = {"id": "v", "prompt_tokens": 10, "split": "eval", "dataset_split": "validation"}
            write_rows(run / "data/train.jsonl", [train])
            write_rows(run / "data/monitor.jsonl", [val])
            cfg = {
                "run_dir": str(run),
                "base_dir": str(base),
                "production_config": str(prod),
                "production_service": None,
                "gpu_uuids": ["gpu0", "gpu1"],
                "mentor": {},
                "validation_monitor_rows": 1,
                "max_new_tokens": 320,
                "learning_rate": 5e-6,
                "python": "python",
                "target_model": "target.gguf",
                "initial_draft": "parent.gguf",
                "batch_size": 4,
                "training_context": 384,
                "output_name": "draft-candidate-F16.gguf",
                "model_name": "ModelImprovementLab-candidate",
            }
            if limit is not None:
                cfg["max_epochs"] = limit
            plan = {
                "shards": [{"key": "e00-s0000", "path": str(run / "data/train.jsonl")}],
                "steps_per_epoch_estimate": 1,
                "max_prompt_tokens": 10,
                "production_config_sha256": hashlib.sha256(prod.read_bytes()).hexdigest(),
            }
            atomic_json(run / "plan.json", plan)
            config = base / "config.json"
            config.write_text(json.dumps(cfg))
            if strategy:
                cfg["mentor"]["model"] = "deepseek-flash"
                config.write_text(json.dumps(cfg))
                atomic_json(
                    run / "evidence/strategy-control.json",
                    {
                        "review_key": "strategy-test",
                        "plan_sha256": hashlib.sha256((run / "plan.json").read_bytes()).hexdigest(),
                    },
                )
                atomic_json(
                    run / "evidence/mentor/strategy-test.json",
                    {
                        "request_id": "provider-strategy-id",
                        "model_requested": "deepseek-flash",
                        "evidence": {"phase": "strategy_review", "current_step": 0},
                        "decision": strategy,
                    },
                )
            metric = {
                "baseline": {"loss": 2.0, "prefix_tokens": 2.0},
                "last": {"loss": 3.0, "prefix_tokens": 1.0},
                "best": {"loss": 2.0, "prefix_tokens": 2.0},
                "train_examples": 1,
                "peak_memory_gib": 1,
                "steps": 1,
                "best_step": 1,
                "learning_rate": 5e-6,
                "stage_key": "e0000-s0000",
            }
            if resumed:
                atomic_json(run / "evidence/run-state.json", resumed)
                atomic_json(run / "outputs/training.json", metric)
            train_commands = []

            def fake_stage(command, env, log, config):
                if "benchmark" in command:
                    atomic_json(run / "outputs/benchmark-summary.json", {"candidate_installed": False})
                else:
                    train_commands.append(command)
                    atomic_json(
                        run / "outputs/training.json",
                        {
                            **metric,
                            "stage_key": command[command.index("--stage-key") + 1],
                            "learning_rate": float(command[command.index("--learning-rate") + 1]),
                        },
                    )

            def fake_capture(config, prompts, directory, env, log):
                from scripts.coach_core.files import read_rows

                Path(directory).mkdir(exist_ok=True)
                return list(read_rows(prompts))

            with (
                patch.object(sys, "argv", ["run.py", "--config", str(config), "--start-authorized"]),
                patch("scripts.orchestration.controller.load_config", return_value=cfg),
                patch("scripts.orchestration.session.signal.signal"),
                patch("scripts.orchestration.controller.check", return_value=(cfg, plan, {})),
                patch("scripts.orchestration.session.subprocess.run", return_value=SimpleNamespace(returncode=0)),
                patch("scripts.orchestration.runtime.local_api", side_effect=lambda path, **kwargs: [] if path == "/slots" else {"status": "ok"}),
                patch("scripts.orchestration.runtime.capture", side_effect=fake_capture),
                patch("scripts.orchestration.runtime.run_stage", side_effect=fake_stage),
                patch("scripts.mentoring.review.review", side_effect=decisions),
            ):
                supervisor.main()
            return json.loads((run / "evidence/run-state.json").read_text()), train_commands

    def decision(self, action="continue", rate=None):
        return {"action": action, "reason": "mentor evidence", "learning_rate": rate}

    def resumed(self, finished=False):
        return {
            "status": "stopped",
            "epoch": 0,
            "next_shard": 1,
            "committed": ["e0000-s0000"],
            "learning_rate": 1e-6,
            "history": [],
            "training_finished": finished,
        }

    def test_mentor_requests_second_epoch_and_changes_rate_then_finishes(self):
        state, commands = self.exercise(
            [
                self.decision(),
                self.decision(rate=1e-6),
                self.decision(rate=2e-6),
                self.decision("stop"),
                self.decision(),
            ]
        )
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[1][commands[1].index("--learning-rate") + 1], "2e-06")
        self.assertEqual(state["epoch"], 1)
        self.assertTrue(state["training_finished"])
        self.assertEqual(state["status"], "completed")
        self.assertTrue(state["production_restored"])

    def test_provider_strategy_changes_next_training_rate_without_repeating_resume_review(self):
        state, commands = self.exercise([self.decision("stop"), self.decision()], strategy=self.decision(rate=2.5e-6))
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][commands[0].index("--learning-rate") + 1], "2.5e-06")
        self.assertEqual(state["applied_strategy_requests"], ["provider-strategy-id"])
        self.assertEqual(state["last_strategy_review"]["decision"]["learning_rate"], 2.5e-6)

    def test_resume_at_epoch_boundary_does_not_repeat_committed_training(self):
        state, commands = self.exercise([self.decision(), self.decision("stop"), self.decision()], self.resumed())
        self.assertEqual(commands, [])
        self.assertTrue(state["training_finished"])
        self.assertEqual(state["status"], "completed")

    def test_configured_epoch_limit_stops_after_one_pass(self):
        state, commands = self.exercise([self.decision(), self.decision(), self.decision()], limit=1)
        self.assertEqual(len(commands), 1)
        self.assertEqual(state["training_stop_reason"], "Configured max_epochs reached")

    def test_mentor_can_stop_before_any_gpu_work(self):
        state, commands = self.exercise([self.decision("stop")])
        self.assertEqual(commands, [])
        self.assertEqual(state["stage"], "mentor_finished_before_training")

    def test_resume_after_mentor_finished_runs_benchmark_only(self):
        state, commands = self.exercise([self.decision()], self.resumed(True))
        self.assertEqual(commands, [])
        self.assertEqual(state["status"], "completed")


if __name__ == "__main__":
    unittest.main()
