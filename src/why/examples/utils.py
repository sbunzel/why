import json
from typing import Any, Dict

from ..utils import get_root_dir

__all__ = ["read_data_config"]


def read_data_config(name: str) -> Dict[str, Any]:
    root_path = get_root_dir()
    config_path = (root_path / "resources" / "config" / name).resolve()
    with open(config_path, mode="r") as f:
        config = json.load(f)
    return config
