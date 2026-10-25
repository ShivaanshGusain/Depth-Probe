"""Tiny randomly initialized models built from configs: no downloads needed."""

import pytest
import torch
from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig,
                          LlamaForCausalLM, MixtralConfig, MixtralForCausalLM, Qwen3Config, Qwen3ForCausalLM,
                          Qwen3MoeConfig, Qwen3MoeForCausalLM)

VOCAB = 97
WIDTH = 32
# Architectures the depth measurements (H1) must support; finetune tests keep the original three.
H1_KINDS = ["gpt2", "llama", "neox", "qwen3", "qwen3_moe", "mixtral"]


def tiny_config(kind, **overrides):
    shared = dict(vocab_size=VOCAB, hidden_size=WIDTH, intermediate_size=64, num_hidden_layers=3,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    if kind == "qwen3":
        return Qwen3Config(**shared, head_dim=8, **overrides)
    if kind == "qwen3_moe":
        return Qwen3MoeConfig(**shared, head_dim=8, num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
                              decoder_sparse_step=1, mlp_only_layers=[], **overrides)
    if kind == "mixtral":
        return MixtralConfig(**shared, num_local_experts=4, num_experts_per_tok=2, **overrides)
    raise ValueError(kind)


MODEL_CLASSES = {"qwen3": Qwen3ForCausalLM, "qwen3_moe": Qwen3MoeForCausalLM, "mixtral": MixtralForCausalLM}


def tiny(kind):
    torch.manual_seed(0)
    if kind == "gpt2":
        model = GPT2LMHeadModel(GPT2Config(vocab_size=VOCAB, n_positions=64, n_embd=WIDTH, n_layer=3, n_head=4))
    elif kind == "llama":
        model = LlamaForCausalLM(LlamaConfig(vocab_size=VOCAB, hidden_size=WIDTH, intermediate_size=64,
                                             num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                                             max_position_embeddings=64))
    elif kind == "neox":
        model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=VOCAB, hidden_size=WIDTH, intermediate_size=64,
                                                 num_hidden_layers=3, num_attention_heads=4,
                                                 max_position_embeddings=64))
    elif kind in MODEL_CLASSES:
        model = MODEL_CLASSES[kind](tiny_config(kind))
    else:
        raise ValueError(kind)
    return model.eval().requires_grad_(False)


@pytest.fixture(params=["gpt2", "llama", "neox"])
def model(request):
    return tiny(request.param)
