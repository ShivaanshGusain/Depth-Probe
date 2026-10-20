"""Model loading and residual-stream access.

Hidden states are captured with our own hooks instead of `output_hidden_states`, because
whether the last entry of that tuple is already final-normalized differs between models and
transformers versions. The same hooks are where noise injection (E6/E7) will go.

State indexing used everywhere: state 0 = residual entering block 0 (embeddings),
state l = residual leaving block l (1-based), state L = last block output, before final norm.
"""

import copy
from dataclasses import dataclass

import numpy as np
import torch
import transformers
from packaging.version import Version
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
# Readouts that are not plain "final norm, then linear head" would decode every layer wrongly.
UNSUPPORTED_CONFIG_KEYS = ("final_logit_softcapping", "logit_scale", "logits_scaling")


@dataclass
class Parts:
    family: str
    container: nn.Module
    blocks_attr: str
    norm: nn.Module
    head: nn.Module

    @property
    def blocks(self):
        return getattr(self.container, self.blocks_attr)


def find_parts(model):
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return Parts("gpt2", model.transformer, "h", model.transformer.ln_f, model.lm_head)
    if hasattr(model, "gpt_neox"):
        return Parts("gpt_neox", model.gpt_neox, "layers", model.gpt_neox.final_layer_norm, model.embed_out)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers") and hasattr(inner, "norm"):
        # Llama, Mistral, Qwen2/3, OLMo-2 and similar decoder layouts.
        return Parts("llama", inner, "layers", inner.norm, model.lm_head)
    raise ValueError(f"Unsupported architecture: {type(model).__name__}")


def n_blocks(model):
    return len(find_parts(model).blocks)


def hidden_width(model):
    return find_parts(model).norm.weight.shape[-1]


def max_context(model):
    config = model.config
    return getattr(config, "n_positions", None) or getattr(config, "max_position_embeddings")


def resolve_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_tokenizer(path):
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def load_model(path, *, dtype="float32", init="pretrained", init_seed=0, device="cpu", attn=None):
    torch_dtype = DTYPES[dtype]
    kwargs = {"attn_implementation": attn} if attn else {}
    if init == "pretrained":
        key = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, **{key: torch_dtype}, **kwargs)
    elif init == "random":
        # Control E5(i): same architecture, no learning. Seeded on CPU so every machine agrees.
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            model = AutoModelForCausalLM.from_config(config, **kwargs)
    else:
        raise ValueError(f"Unknown init: {init}")
    for key in UNSUPPORTED_CONFIG_KEYS:
        if getattr(model.config, key, None) not in (None, 1, 1.0):
            raise ValueError(f"{key} is set; the readout would not match the model's own head")
    find_parts(model)
    model = model.to(device=device, dtype=torch_dtype).eval()
    model.requires_grad_(False)
    return model


def shuffle_permutation(count, seed, mode):
    rng = np.random.default_rng(seed)
    if mode == "all":
        return [int(i) for i in rng.permutation(count)]
    if mode == "middle":
        # First and last blocks are unusually important (Lad et al. 2024), so keep them in place.
        return [0, *(int(i) + 1 for i in rng.permutation(count - 2)), count - 1]
    raise ValueError(f"Unknown shuffle mode: {mode}")


def permute_blocks(model, permutation):
    # Control E5(ii). Only valid with use_cache=False: attention layers keep their original layer_idx.
    parts = find_parts(model)
    blocks = list(parts.blocks)
    if sorted(permutation) != list(range(len(blocks))):
        raise ValueError("Not a permutation of the blocks")
    setattr(parts.container, parts.blocks_attr, nn.ModuleList(blocks[i] for i in permutation))


class ResidualCapture:
    """Records the residual stream entering block 0 and leaving every block."""

    def __init__(self, model):
        self.blocks = find_parts(model).blocks
        self.states = []
        self.handles = []

    def __enter__(self):
        self.states = []
        self.handles.append(self.blocks[0].register_forward_pre_hook(self._entering, with_kwargs=True))
        for block in self.blocks:
            self.handles.append(block.register_forward_hook(self._leaving))
        return self

    def __exit__(self, *exception):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _entering(self, module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        self.states = [hidden.detach().clone()]

    def _leaving(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.states.append(hidden.detach().clone())


@torch.no_grad()
def forward_states(model, input_ids, **kwargs):
    with ResidualCapture(model) as capture:
        output = model(input_ids=input_ids, use_cache=False, **kwargs)
    if len(capture.states) != n_blocks(model) + 1:
        raise RuntimeError(f"Captured {len(capture.states)} states for {n_blocks(model)} blocks")
    return capture.states, output


class Readout(nn.Module):
    """The model's own final norm and unembedding, as frozen float32 copies, applied to any state."""

    def __init__(self, model):
        super().__init__()
        parts = find_parts(model)
        self.norm = copy.deepcopy(parts.norm).float().requires_grad_(False)
        self.head = copy.deepcopy(parts.head).float().requires_grad_(False)

    def forward(self, hidden):
        return self.head(self.norm(hidden.float()))
