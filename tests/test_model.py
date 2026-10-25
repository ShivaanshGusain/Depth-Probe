import pytest
import torch

from conftest import H1_KINDS, MODEL_CLASSES, VOCAB, WIDTH, tiny, tiny_config
from depthprobe.model import (Readout, find_parts, forward_states, load_model, permute_blocks,
                              shuffle_permutation)


@pytest.fixture(params=H1_KINDS)
def model(request):
    return tiny(request.param)


def ids(batch=2, length=16):
    return torch.randint(0, VOCAB, (batch, length), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("kind", ["llama", "qwen3_moe"])
def test_readout_in_model_dtype_reproduces_bfloat16_logits(kind):
    model = tiny(kind).to(torch.bfloat16)
    states, output = forward_states(model, ids())
    logits = Readout(model, dtype=torch.bfloat16)(states[-1])
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, output.logits.float(), rtol=0, atol=0)


def test_random_init_is_constructed_in_requested_dtype(tmp_path, monkeypatch):
    import depthprobe.model

    tiny("qwen3").config.save_pretrained(tmp_path)
    original = depthprobe.model.AutoModelForCausalLM.from_config
    built = {}

    def spy(config, **kwargs):
        model = original(config, **kwargs)
        built["default"] = torch.get_default_dtype()
        built["parameters"] = {parameter.dtype for parameter in model.parameters()}
        return model

    monkeypatch.setattr(depthprobe.model.AutoModelForCausalLM, "from_config", spy)
    for name, dtype in (("bfloat16", torch.bfloat16), ("float32", torch.float32)):
        load_model(tmp_path, dtype=name, init="random", init_seed=1)
        # Checked at construction, before any later cast: a 30B model must never exist in float32.
        assert built["parameters"] == {dtype} and built["default"] == dtype
        assert torch.get_default_dtype() == torch.float32


def test_bfloat16_random_init_is_reproducible(tmp_path):
    tiny("qwen3").config.save_pretrained(tmp_path)
    first = load_model(tmp_path, dtype="bfloat16", init="random", init_seed=1)
    second = load_model(tmp_path, dtype="bfloat16", init="random", init_seed=1)
    weight = "model.layers.0.mlp.up_proj.weight"
    torch.testing.assert_close(first.state_dict()[weight], second.state_dict()[weight], rtol=0, atol=0)


def test_random_init_restores_default_dtype_on_failure(tmp_path, monkeypatch):
    import depthprobe.model

    tiny("qwen3").config.save_pretrained(tmp_path)

    def broken(config, **kwargs):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(depthprobe.model.AutoModelForCausalLM, "from_config", broken)
    try:
        with pytest.raises(RuntimeError):
            load_model(tmp_path, dtype="bfloat16", init="random")
        observed = torch.get_default_dtype()
    finally:
        torch.set_default_dtype(torch.float32)  # keep a regression here from breaking unrelated tests
    assert observed == torch.float32


@pytest.mark.parametrize("init", ["pretrained", "random"])
@pytest.mark.parametrize("kind", ["qwen3", "qwen3_moe", "mixtral"])
def test_bfloat16_load_keeps_rope_frequencies_in_float32(tmp_path, kind, init):
    tiny(kind).save_pretrained(tmp_path)
    loaded = load_model(tmp_path, dtype="bfloat16", init=init)
    reference = load_model(tmp_path, dtype="float32", init=init)
    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.bfloat16}
    frequencies = {name: buffer for name, buffer in loaded.named_buffers() if name.endswith("inv_freq")}
    expected = {name: buffer for name, buffer in reference.named_buffers() if name.endswith("inv_freq")}
    assert frequencies and frequencies.keys() == expected.keys()
    for name, buffer in frequencies.items():
        assert buffer.dtype == torch.float32, name
        torch.testing.assert_close(buffer, expected[name], rtol=0, atol=0)


def test_moe_router_loss_is_disabled_at_load(tmp_path):
    config = tiny_config("qwen3_moe", output_router_logits=True, router_aux_loss_coef=1.0)
    MODEL_CLASSES["qwen3_moe"](config).save_pretrained(tmp_path)
    assert load_model(tmp_path).config.output_router_logits is False


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
