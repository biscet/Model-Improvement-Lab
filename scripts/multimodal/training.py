"""Bounded training using Transformers checkpoint and optimizer recovery."""

import fcntl
import json
import time
from pathlib import Path

from scripts.coach_core.files import atomic_json
from scripts.multimodal.configuration import dataset_identity, read_dataset
from scripts.multimodal.models import Collator, load_model


def train(cfg, resume=False):
    run = Path(cfg['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another worker owns this run directory') from None
        return train_locked(cfg, run, resume)


def train_locked(cfg, run, resume):
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    rows = read_dataset(cfg)
    validation_rows = read_dataset({**cfg, 'dataset': cfg['validation_dataset']}) if cfg.get('validation_dataset') else None
    identity = {'task': cfg['task'], 'model': cfg['model'], 'revision': cfg['revision'],
                'dataset': dataset_identity(cfg, rows),
                'recipe': {k: cfg[k] for k in ('batch_size', 'gradient_accumulation_steps',
                                             'lora_rank', 'learning_rate', 'seed', 'dtype')}}
    if validation_rows is not None:
        validation_identity = dataset_identity({**cfg, 'dataset': cfg['validation_dataset']}, validation_rows)
        if set(identity['dataset']['media_sha256']) & set(validation_identity['media_sha256']):
            raise ValueError('Training and validation media overlap')
        identity['validation_dataset'] = validation_identity
    if cfg.get('warmup_steps'):
        identity['recipe']['warmup_steps'] = cfg['warmup_steps']
    manifest_path = run / 'experiment.json'
    state_path = run / 'run_state.json'
    checkpoint = None
    if manifest_path.exists():
        if not resume:
            raise ValueError('Run already exists; use --resume or a new run_dir')
        if json.loads(manifest_path.read_text()) != identity:
            raise ValueError('Model, training recipe or dataset changed since the run began')
        checkpoint = get_last_checkpoint(str(run))
        if checkpoint is None:
            raise ValueError('No complete checkpoint exists to resume; use a new run_dir')
    elif resume:
        raise ValueError('Run does not exist')
    elapsed_before = 0
    if state_path.exists():
        elapsed_before = json.loads(state_path.read_text()).get('elapsed_seconds', 0)
    if elapsed_before >= cfg['max_seconds']:
        raise ValueError('The cumulative training time budget is exhausted')
    atomic_json(manifest_path, identity)
    set_seed(cfg['seed'])
    model, processor = load_model(cfg, training=True)
    if cfg['task'] != 'audio':
        model.config.use_cache = False
        model.enable_input_require_grads()
    started = time.monotonic()

    class BudgetCallback(TrainerCallback):
        optimizer_updates = 0
        skipped_updates = 0

        def write_state(self, state, status):
            atomic_json(state_path, {
                'status': status, 'step': state.global_step,
                'optimizer_updates_this_invocation': self.optimizer_updates,
                'skipped_updates_this_invocation': self.skipped_updates,
                'elapsed_seconds': elapsed_before + time.monotonic() - started,
                'max_seconds': cfg['max_seconds'],
            })

        def on_step_end(self, args, state, control, **kwargs):
            if trainer.accelerator.optimizer_step_was_skipped:
                self.skipped_updates += 1
            else:
                self.optimizer_updates += 1
            self.write_state(state, 'training')
            if elapsed_before + time.monotonic() - started >= cfg['max_seconds']:
                control.should_training_stop = True
                control.should_save = True
            return control

    class MultimodalTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            if cfg['task'] == 'audio':
                inputs = {**inputs, 'return_loss': True}
            outputs = model(**inputs)
            if not torch.isfinite(outputs.loss):
                raise RuntimeError('Nonfinite training loss')
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    args = TrainingArguments(
        output_dir=str(run), max_steps=cfg['max_steps'],
        per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        learning_rate=cfg['learning_rate'],
        lr_scheduler_type='constant_with_warmup' if cfg.get('warmup_steps') else 'constant',
        warmup_steps=cfg.get('warmup_steps', 0),
        save_strategy='steps', save_steps=cfg.get('save_steps', 25), save_total_limit=2,
        eval_strategy='steps' if validation_rows is not None else 'no',
        eval_steps=cfg.get('eval_steps', cfg.get('save_steps', 25)),
        per_device_eval_batch_size=1, prediction_loss_only=True,
        load_best_model_at_end=validation_rows is not None,
        metric_for_best_model='eval_loss' if validation_rows is not None else None,
        greater_is_better=False,
        logging_steps=1, report_to=[], remove_unused_columns=False,
        dataloader_drop_last=cfg['task'] == 'audio', dataloader_pin_memory=False,
        fp16=cfg['dtype'] == 'float16', bf16=cfg['dtype'] == 'bfloat16',
        use_cpu=cfg['device'] == 'cpu',
        gradient_checkpointing=cfg['task'] != 'audio',
        gradient_checkpointing_kwargs={'use_reentrant': False},
        seed=cfg['seed'], data_seed=cfg['seed'], optim='adamw_torch',
    )
    callback = BudgetCallback()
    trainer = MultimodalTrainer(model=model, args=args, train_dataset=rows, eval_dataset=validation_rows,
                               data_collator=Collator(cfg, processor), callbacks=[callback])
    trainer.model_accepts_loss_kwargs = False
    if trainer.accelerator.scaler is not None:
        # Large initial AMP scales can overflow on the first multimodal batch.
        # A resumed checkpoint restores its own scaler state during train().
        scaler = trainer.accelerator.scaler
        scaler.load_state_dict({**scaler.state_dict(), 'scale': 128.0})
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        if callback.optimizer_updates == 0:
            raise RuntimeError('No optimizer update completed; AMP skipped every step')
        if any(not torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad):
            raise RuntimeError('Nonfinite trained parameters; candidate was not exported')
        # Trainer saves the last step, including optimizer and RNG, at normal completion.
        destination = run / 'candidate'
        trainer.save_model(str(destination))
        processor.save_pretrained(destination)
        callback.write_state(trainer.state, 'completed')
        atomic_json(destination / 'training_summary.json', {
            **identity, 'steps': trainer.state.global_step,
            'optimizer_updates_this_invocation': callback.optimizer_updates,
            'skipped_updates_this_invocation': callback.skipped_updates,
            'selected_checkpoint': Path(trainer.state.best_model_checkpoint).name if trainer.state.best_model_checkpoint else None,
            'best_validation_loss': trainer.state.best_metric,
            'elapsed_seconds': elapsed_before + time.monotonic() - started,
            'method': 'contrastive-projections' if cfg['task'] == 'audio' else 'lora',
        })
        return json.loads(state_path.read_text())
    except BaseException:
        callback.write_state(trainer.state, 'interrupted')
        raise
