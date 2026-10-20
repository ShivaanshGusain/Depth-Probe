"""Tiny randomly initialized models built from configs: no downloads needed."""

import pytest
import torch
from transformers import (GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM, LlamaConfig,
                          LlamaForCausalLM)

VOCAB = 97
WIDTH = 32


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
    else:
        raise ValueError(kind)
    return model.eval().requires_grad_(False)


@pytest.fixture(params=["gpt2", "llama", "neox"])
def model(request):
    return tiny(request.param)
