"""Project-level paths, overridable by the public stage configuration."""

import os
from pathlib import Path
from msi import REPO_ROOT as PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = Path(os.environ.get(
    "MSI_EXTERNAL_DIR", str(DATA_DIR / "external")
))
CORPUS_PATH = Path(os.environ.get(
    "MSI_SKILL_CORPUS", str(DATA_DIR / "protocol" / "skills.json")
))
RESULTS_DIR = Path(os.environ.get("MSI_RESULTS_DIR", str(PROJECT_ROOT / "results")))
INSTANCES_DIR = Path(os.environ.get(
    "MSI_INSTANCES_DIR",
    str(PROJECT_ROOT / "SR-Agents" / "data" / "bench" / "instances"),
))
