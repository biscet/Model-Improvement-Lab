#!/usr/bin/env python3
"""Run the selected Model Improvement Lab command without importing the ML backend eagerly."""

import argparse
import importlib
import sys

from scripts.coach_core.constants.commands import COMMANDS


def main():
    parser = argparse.ArgumentParser(
        description="Model Improvement Lab runs, fine-tunes and packages vision, speech, music and draft models.",
    )
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = parser.parse_args()

    module_name = "scripts.commands." + COMMANDS[args.command]
    command = importlib.import_module(module_name)
    sys.argv = [f"{sys.argv[0]} {args.command}", *args.arguments]
    command.main()


if __name__ == "__main__":
    main()
