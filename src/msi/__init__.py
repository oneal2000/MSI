"""MSI: synthetic skill data, anchors, LoRA training and evaluation."""

from pathlib import Path

__version__ = "1.0.0"

# Checkout root used only for zero-install execution and bundled configs. Runtime
# artifacts are always resolved from the user config, never from this directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
