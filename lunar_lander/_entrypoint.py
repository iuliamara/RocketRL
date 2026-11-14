"""Resolve package imports when a CLI is launched by its original file path."""

import sys
from pathlib import Path


def package_name():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    return "lunar_lander"
