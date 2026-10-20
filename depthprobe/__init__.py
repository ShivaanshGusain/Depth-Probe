"""Layer-wise probes for the depth-as-denoising project."""

import os

# Never download: models and data are fetched by the user and passed as local paths.
for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_name, "1")

__version__ = "0.1.0"
