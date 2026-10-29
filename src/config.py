from pathlib import Path
import yaml

def load_params(path: str = "params.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())
