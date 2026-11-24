"""Stage reproducible Hugging Face packages with original model attribution."""

import json
import shutil
import tempfile
from pathlib import Path

from scripts.coach_core.files import atomic_json, digest


def package(cfg, output, trained=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('Choose a new package output directory')
    source = Path(cfg['model_dir'])
    candidate = Path(cfg['run_dir']) / 'candidate'
    summary = None
    if trained:
        summary = json.loads((candidate / 'training_summary.json').read_text())
        if (summary['model'], summary['revision'], summary['task']) != (cfg['model'], cfg['revision'], cfg['task']):
            raise ValueError('Candidate belongs to a different model or task')
        source = candidate
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '-', dir=output.parent))
    try:
        for path in source.iterdir():
            if path.is_file() and path.suffix in ('.json', '.jinja', '.txt', '.safetensors', '.bin'):
                if path.name != 'training_summary.json':
                    shutil.copy2(path, staging / path.name)
        base = Path(cfg['model_dir'])
        if (base / 'README.md').exists():
            shutil.copy2(base / 'README.md', staging / 'UPSTREAM_README.md')
        for path in base.iterdir():
            if path.is_file() and path.name.startswith(('LICENSE', 'NOTICE')):
                name = 'UPSTREAM_' + path.name if path.name.startswith('NOTICE') else path.name
                shutil.copy2(path, staging / name)
        for path in base.rglob('*License*.docx'):
            target = staging / path.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        if cfg['license'] == 'apache-2.0' and not any(p.name.startswith('LICENSE') for p in staging.iterdir()):
            shutil.copy2(Path(__file__).resolve().parents[2] / 'LICENSE', staging / 'LICENSE')
        adapter_path = staging / 'adapter_config.json'
        if adapter_path.exists():
            adapter = json.loads(adapter_path.read_text())
            adapter['base_model_name_or_path'] = cfg['model']
            adapter['revision'] = cfg['revision']
            atomic_json(adapter_path, adapter)
        tag = {'vision': 'image-text-to-text', 'speech': 'automatic-speech-recognition',
               'audio': 'feature-extraction', 'music': 'audio-text-to-text'}[cfg['task']]
        provenance = {'publisher': cfg.get('publisher', 'biscet'), 'base_model': cfg['model'],
                      'revision': cfg['revision'], 'license': cfg['license'],
                      'fine_tuned': trained, 'training': summary}
        atomic_json(staging / 'PROVENANCE.json', provenance)
        status = 'Fine-tuned with Model Improvement Lab.' if trained else 'Upstream weights packaged and configured with Model Improvement Lab.'
        license_fields = ''
        if cfg.get('license_name'):
            license_fields = f"license_name: {cfg['license_name']}\nlicense_link: {cfg['license_link']}\n"
        card = f"""---
license: {cfg['license']}
{license_fields}\
base_model: {cfg['model']}
pipeline_tag: {tag}
library_name: {'peft' if adapter_path.exists() else 'transformers'}
tags:
- model-improvement-lab
- {cfg['task']}
---

# {cfg['publish_name']}

{status}

Prepared by {cfg.get('publisher', 'biscet')}. Original model authors retain attribution.
The source checkpoint is `{cfg['model']}` at revision `{cfg['revision']}`.
Original documentation is included in UPSTREAM_README.md.
{cfg.get('license_note', '')}

This package is a companion model for {cfg['task']} processing. Model Improvement Lab can
run it independently and supply its results to a language model.

Use the original Transformers loading instructions for base packages.
For LoRA packages, load the pinned base model and attach this package with
`PeftModel.from_pretrained`. The processor is included.

Validation records describe individual checks. Task-level accuracy and
improvement require evaluation on representative held-out examples.
"""
        (staging / 'README.md').write_text(card)
        (staging / 'NOTICE').write_text(
            f"Packaged by {cfg.get('publisher', 'biscet')} using Model Improvement Lab.\n"
            f"Original model: {cfg['model']}\nRevision: {cfg['revision']}\n"
            "See UPSTREAM_README.md and the included license documents for attribution and terms.\n"
        )
        atomic_json(staging / 'FILES.json', {
            str(p.relative_to(staging)): {'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(staging.rglob('*')) if p.is_file()
        })
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return {'package': str(output), 'fine_tuned': trained}
