import json
from pathlib import Path
from typing import Any, Dict

__all__ = ["get_root_dir", "read_config"]


def get_root_dir():
    src_paths = [p for p in Path(__file__).parents if str(p.resolve()).endswith("src")]
    assert len(src_paths) == 1, "There must be only src directory in the directory tree"
    return src_paths[0].parent


def read_config(name: str) -> Dict[str, Any]:
    root_path = get_root_dir()
    config_path = (root_path / "resources" / "config" / name).resolve()
    with open(config_path, mode="r") as f:
        config = json.load(f)
    return config
