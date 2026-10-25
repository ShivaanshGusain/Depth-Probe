"""What produced a result: code, libraries, hardware, and model files."""

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def source_sha256():
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*.py")):
        digest.update(path.relative_to(PACKAGE).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_fingerprint(path):
    # File names and sizes, not a content hash of the weights (that would take minutes for 8B models).
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts)
    config = root / "config.json"
    return {
        "path": str(root.resolve()),
        "files": [{"name": p.relative_to(root).as_posix(), "bytes": p.stat().st_size} for p in files],
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest() if config.exists() else None,
    }


def environment():
    import numpy
    import torch
    import transformers

    cuda = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": numpy.__version__,
        "device": torch.cuda.get_device_name() if cuda else (platform.processor() or "cpu"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "source_sha256": source_sha256(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _finite(value):
    """NaN and infinities become null: strict JSON has no such numbers (Python would write NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_finite(data), indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)
