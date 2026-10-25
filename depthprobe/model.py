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
    # The output head's attribute name varies across versions (GPT-NeoX: embed_out, later lm_head).
    head = model.get_output_embeddings()
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return Parts("gpt2", model.transformer, "h", model.transformer.ln_f, head)
    if hasattr(model, "gpt_neox"):
        return Parts("gpt_neox", model.gpt_neox, "layers", model.gpt_neox.final_layer_norm, head)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers") and hasattr(inner, "norm"):
        # Llama, Mistral, Qwen2/3, OLMo-2 and similar decoder layouts.
        return Parts("llama", inner, "layers", inner.norm, head)
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
        # Built directly in the requested dtype: a 30B model initialized in float32 first would need ~120 GB.
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        config.dtype = torch_dtype  # from_config falls back to the config's dtype (transformers >= 4.56)
        previous = torch.get_default_dtype()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            torch.set_default_dtype(torch_dtype)
            try:
                model = AutoModelForCausalLM.from_config(config, **kwargs)
            finally:
                torch.set_default_dtype(previous)
    else:
        raise ValueError(f"Unknown init: {init}")
    for key in UNSUPPORTED_CONFIG_KEYS:
        if getattr(model.config, key, None) not in (None, 1, 1.0):
            raise ValueError(f"{key} is set; the readout would not match the model's own head")
    # Mixture-of-experts: with router logits on, the library's loss adds a load-balancing term, so it is no
    # longer plain next-token loss and the E0 label check would fail. Routing itself is unaffected.
    if getattr(model.config, "output_router_logits", False):
        model.config.output_router_logits = False
    find_parts(model)
    # Both loaders already build the weights in torch_dtype; only stray parameters are cast (on CPU, before the
    # move). A whole-module .to(dtype) would also round float32 buffers such as RoPE inv_freq and shift positions.
    for parameter in model.parameters():
        if parameter.dtype != torch_dtype:
            parameter.data = parameter.data.to(torch_dtype)
    model = model.to(device=device).eval()
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
    """The model's own final norm and unembedding, as frozen copies, applied to any state.

    dtype=float32 decodes every state in full precision. Passing the model's own dtype (e.g. bfloat16)
    repeats the arithmetic of its native head, so the last state reproduces the model's logits exactly;
    a float32 readout of a bfloat16 model differs from them by bfloat16 rounding. Logits are float32 either way.
    """

    def __init__(self, model, dtype=torch.float32):
        super().__init__()
        parts = find_parts(model)
        self.dtype = dtype
        self.norm = copy.deepcopy(parts.norm).to(dtype).requires_grad_(False)
        self.head = copy.deepcopy(parts.head).to(dtype).requires_grad_(False)

    def forward(self, hidden):
        return self.head(self.norm(hidden.to(self.dtype))).float()
