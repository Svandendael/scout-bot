"""Scout: a monthly ETF ranking bot for a Belgian retail investor."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_yaml(name: str) -> dict:
    with open(ROOT / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    return load_yaml("config.yaml")


def load_universe() -> dict:
    u = load_yaml("universe.yaml")
    u["by_id"] = {e["id"]: e for e in u["etfs"]}
    return u
