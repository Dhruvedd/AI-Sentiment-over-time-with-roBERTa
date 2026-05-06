"""Config loader. Single source of truth for paths and parameters."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml from project root."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve paths relative to project root
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(PROJECT_ROOT / val)
    return cfg


def gcp_project_id(cfg: dict | None = None) -> str:
    """
    Resolve GCP project ID. Precedence:
      1. GCP_PROJECT_ID environment variable
      2. config.yaml gcp_project_id
      3. raise
    """
    pid = os.getenv("GCP_PROJECT_ID")
    if pid:
        return pid
    if cfg is None:
        cfg = load_config()
    pid = cfg.get("gcp_project_id")
    if not pid or pid == "REPLACE_ME":
        raise RuntimeError(
            "GCP project ID not set. Either:\n"
            "  - set GCP_PROJECT_ID in .env, or\n"
            "  - put it in config.yaml under gcp_project_id, or\n"
            "  - pass --project on the CLI."
        )
    return pid
