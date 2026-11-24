"""Portable experiment recipes and JSONL media datasets."""

import json
from pathlib import Path

from scripts.coach_core.files import digest


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg['task'] not in ('vision', 'speech', 'audio', 'music'):
        raise ValueError('task must be vision, speech, audio or music')
    for key in ('model_dir', 'run_dir', 'dataset', 'validation_dataset'):
        if cfg.get(key):
            cfg[key] = str((path.parent / Path(cfg[key]).expanduser()).resolve())
    if len(cfg['revision']) != 40:
        raise ValueError('Pin revision to a full Hugging Face commit')
    if cfg.get('max_seconds', 10800) <= 0:
        raise ValueError('max_seconds must be positive')
    cfg.setdefault('device', 'cuda:0')
    cfg.setdefault('dtype', 'float16')
    cfg.setdefault('learning_rate', 0.00002)
    cfg.setdefault('batch_size', 2 if cfg['task'] == 'audio' else 1)
    cfg.setdefault('gradient_accumulation_steps', 4)
    cfg.setdefault('max_steps', 100000)
    cfg.setdefault('max_seconds', 10800)
    cfg.setdefault('seed', 42)
    cfg.setdefault('lora_rank', 8)
    cfg.setdefault('license', 'apache-2.0')
    if cfg['task'] == 'audio' and cfg['batch_size'] < 2:
        raise ValueError('CLAP contrastive training needs at least two pairs per batch')
    if cfg['task'] != 'audio' and cfg['batch_size'] != 1:
        raise ValueError('Generative recipes currently use batch_size=1 and gradient accumulation')
    for key in ('max_steps', 'batch_size', 'gradient_accumulation_steps', 'lora_rank'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    return cfg


def read_dataset(cfg):
    path = Path(cfg['dataset'])
    rows = []
    media_key = 'image' if cfg['task'] == 'vision' else 'audio'
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError(f'Row {line_number} needs nonempty text')
        media = (path.parent / row[media_key]).resolve()
        if not media.is_file():
            raise ValueError(f'Missing {media_key} at row {line_number}: {media}')
        row[media_key] = str(media)
        if cfg['task'] == 'speech' and not row.get('language'):
            raise ValueError(f'Row {line_number} needs a language, such as Russian')
        rows.append(row)
    if len(rows) < cfg['batch_size']:
        raise ValueError('Dataset must contain at least one complete batch')
    if cfg['task'] == 'audio':
        if len({r['text'] for r in rows}) != len(rows):
            raise ValueError('CLAP pairs need distinct captions to avoid contradictory negatives')
        if len({r['audio'] for r in rows}) != len(rows):
            raise ValueError('CLAP pairs need distinct audio files')
    return rows


def dataset_identity(cfg, rows):
    key = 'image' if cfg['task'] == 'vision' else 'audio'
    return {'jsonl_sha256': digest(cfg['dataset']),
            'media_sha256': [digest(row[key]) for row in rows]}


def download(cfg):
    from huggingface_hub import snapshot_download

    return snapshot_download(
        cfg['model'], revision=cfg['revision'], local_dir=cfg['model_dir'],
        allow_patterns=['*.json', '*.jinja', '*.txt', '*.safetensors', '*.bin', '*.md', 'LICENSE*', 'NOTICE*', '*License*.docx'],
        max_workers=4,
    )


def weights_ready(cfg):
    root = Path(cfg['model_dir'])
    if not (root / 'config.json').is_file():
        return False
    indexes = list(root.glob('*.index.json'))
    if indexes:
        shards = json.loads(indexes[0].read_text())['weight_map'].values()
        return all((root / shard).is_file() for shard in set(shards))
    return any(root.glob('*.safetensors')) or (root / 'pytorch_model.bin').is_file()
