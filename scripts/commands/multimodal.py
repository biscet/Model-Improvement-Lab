"""Prepare, run, fine-tune and package vision, speech and audio models."""

import argparse
import json
from pathlib import Path

from scripts.coach_core.files import atomic_json
from scripts.multimodal.configuration import download, load_config, read_dataset, weights_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['download', 'validate', 'infer', 'train', 'status', 'package'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--prompt')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--trained', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.action == 'download':
        result = {'model_dir': download(cfg), 'revision': cfg['revision']}
    elif args.action == 'validate':
        result = {'records': len(read_dataset(cfg)), 'task': cfg['task']}
    elif args.action == 'infer':
        if args.input is None:
            parser.error('infer requires --input')
        from scripts.multimodal.models import infer
        result = infer(cfg, args.input, args.prompt, args.labels, args.adapter)
        if args.output:
            atomic_json(args.output, result)
    elif args.action == 'train':
        from scripts.multimodal.training import train
        result = train(cfg, args.resume)
    elif args.action == 'package':
        if args.output is None:
            parser.error('package requires --output')
        from scripts.multimodal.artifacts import package
        result = package(cfg, args.output, args.trained)
    else:
        state = Path(cfg['run_dir']) / 'run_state.json'
        result = {'task': cfg['task'], 'model': cfg['model'],
                  'downloaded': weights_ready(cfg),
                  'training': json.loads(state.read_text()) if state.exists() else {'status': 'not_started'}}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
