import torch

from conftest import VOCAB, WIDTH, tiny
from depthprobe.model import (Readout, find_parts, forward_states, load_model, permute_blocks,
                              shuffle_permutation)


def ids(batch=2, length=16):
    return torch.randint(0, VOCAB, (batch, length), generator=torch.Generator().manual_seed(1))


def test_captures_embeddings_and_every_block(model):
    states, _ = forward_states(model, ids())
    assert len(states) == len(find_parts(model).blocks) + 1
    assert all(state.shape == (2, 16, WIDTH) for state in states)


def test_readout_of_last_state_reproduces_native_logits(model):
    states, output = forward_states(model, ids())
    torch.testing.assert_close(Readout(model)(states[-1]), output.logits.float(), rtol=1e-5, atol=1e-5)


def test_first_state_is_gpt2_embedding():
    model = tiny("gpt2")
    tokens = ids()
    states, _ = forward_states(model, tokens)
    embedding = model.transformer.wte(tokens) + model.transformer.wpe(torch.arange(tokens.shape[1]))
    torch.testing.assert_close(states[0], embedding)


def test_permutation_reorders_blocks(model):
    blocks = list(find_parts(model).blocks)
    permute_blocks(model, [2, 0, 1])
    assert [id(block) for block in find_parts(model).blocks] == [id(blocks[i]) for i in (2, 0, 1)]
    states, _ = forward_states(model, ids())
    assert len(states) == 4


def test_identity_permutation_keeps_logits(model):
    before = forward_states(model, ids())[1].logits
    permute_blocks(model, [0, 1, 2])
    torch.testing.assert_close(forward_states(model, ids())[1].logits, before)


def test_shuffle_modes():
    middle = shuffle_permutation(8, 3, "middle")
    assert middle[0] == 0 and middle[-1] == 7 and sorted(middle) == list(range(8))
    assert sorted(shuffle_permutation(8, 3, "all")) == list(range(8))
    assert shuffle_permutation(8, 3, "all") == shuffle_permutation(8, 3, "all")


def test_load_pretrained_and_seeded_random_init(tmp_path):
    reference = tiny("llama")
    reference.save_pretrained(tmp_path)
    loaded = load_model(tmp_path, dtype="float32")
    for (name, a), (_, b) in zip(reference.state_dict().items(), loaded.state_dict().items()):
        torch.testing.assert_close(a, b, msg=name)
    first = load_model(tmp_path, init="random", init_seed=5)
    second = load_model(tmp_path, init="random", init_seed=5)
    other = load_model(tmp_path, init="random", init_seed=6)
    weight = "model.layers.0.mlp.up_proj.weight"
    torch.testing.assert_close(first.state_dict()[weight], second.state_dict()[weight])
    assert not torch.equal(first.state_dict()[weight], other.state_dict()[weight])
