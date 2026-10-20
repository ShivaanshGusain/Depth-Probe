import copy
import csv
import json

import numpy as np
import pytest
import torch

from conftest import VOCAB
from depthprobe.data import Corpus, iterate_batches, make_windows
from depthprobe.finetune import (CachedWindows, cache_training, collate_windows, evaluate_full, l2_deviation,
                               make_segment, parameter_free_rms, segment_step, training_batches,
                               training_postfix, train_segment, validate_range)
from depthprobe.model import Readout, find_parts, forward_states


@pytest.mark.parametrize("first,last", [(1, 1), (2, 2), (2, 3)])
def test_segment_matches_original_and_shares_only_selected_blocks(model, first, last):
    inputs = torch.randint(0, VOCAB, (2, 12))
    states, _ = forward_states(model, inputs)
    parts = find_parts(model)
    original_blocks = list(parts.blocks)
    segment = make_segment(model, first, last)
    actual = segment(inputs_embeds=states[first - 1], use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, states[last], atol=1e-5, rtol=1e-5)
    segment_params = {id(p) for p in segment.parameters() if p.requires_grad}
    expected = {id(p) for block in original_blocks[first - 1:last] for p in block.parameters()}
    assert segment_params == expected
    assert list(find_parts(model).blocks) == original_blocks


@pytest.mark.parametrize("regularizer", ["l2", "kl", "both", "none"])
@pytest.mark.parametrize("l2_space", ["residual", "rms"])
def test_cache_training_and_full_evaluation(model, tmp_path, regularizer, l2_space):
    rng = np.random.default_rng(1)
    corpus = Corpus(rng.integers(0, VOCAB, 40, dtype=np.int64), np.repeat(np.arange(2), 20).astype(np.int32), 2)
    windows = make_windows(40, 16, 8)
    kwargs = dict(batch_size=2, bos_id=0, device="cpu", chunk=7)
    before, before_arrays = evaluate_full(model, corpus, windows, **kwargs)
    readout = Readout(model)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    meta = cache_training(model, corpus, windows, tmp_path, first=2, last=2, **kwargs, log=lambda _: None)
    assert meta["tokens"] == 39
    shards = [torch.load(tmp_path / filename, weights_only=True) for filename in meta["shards"]]
    reference = torch.cat([batch["reference_exit"] for batch in shards]).float()
    stats = torch.load(tmp_path / "activation_stats.pt", weights_only=True)
    torch.testing.assert_close(stats["sample_std"], reference.std(0))
    normalized = parameter_free_rms(reference)
    torch.testing.assert_close(stats["rms"]["mean"], normalized.mean(0))
    torch.testing.assert_close(stats["rms"]["sample_std"], normalized.std(0))
    for batch in shards:
        assert batch["prefix"].device.type == "cpu"
        assert not batch["prefix"].requires_grad
        torch.testing.assert_close(readout(batch["reference_final"]).argmax(-1), batch["native_prediction"])
    segment = make_segment(model, 2, 2)
    history = train_segment(segment, readout, tmp_path, meta, device="cpu", epochs=1, lr=1e-3,
                            strength=0.1, regularizer=regularizer, chunk=7, seed=0, log=lambda _: None,
                            l2_space=l2_space, log_path=tmp_path / "training.csv")
    assert np.isfinite(history[0]["cross_entropy"])
    assert 0 <= history[0]["accuracy"] <= 1
    with (tmp_path / "training.csv").open() as stream:
        logged = list(csv.DictReader(stream))
    if regularizer in ("l2", "both"):
        assert history[0]["l2_space"] == l2_space
        assert np.isfinite(history[0]["max_l2_deviation_std"])
        assert all(np.isfinite(float(row["max_l2_deviation_std"])) for row in logged)
        assert all(np.isfinite(float(row["l2_penalty"])) for row in logged)
    else:
        assert history[0]["l2_space"] is None
        assert history[0]["max_l2_deviation_std"] is None
        assert all(row["max_l2_deviation_std"] == "" for row in logged)
        assert all(row["l2_penalty"] == "" for row in logged)
    if regularizer in ("kl", "both"):
        assert np.isfinite(history[0]["kl_penalty"])
        assert all(np.isfinite(float(row["kl_penalty"])) for row in logged)
    else:
        assert history[0]["kl_penalty"] is None
        assert all(row["kl_penalty"] == "" for row in logged)
    parts = find_parts(model)
    changed = []
    for key, value in model.state_dict().items():
        if not torch.equal(original[key], value):
            changed.append(key)
    assert changed
    selected_ids = {id(p) for p in parts.blocks[1].parameters()}
    for name, parameter in model.named_parameters():
        if id(parameter) not in selected_ids:
            torch.testing.assert_close(parameter, original[name], atol=0, rtol=0)
    assert all(p.grad is None for p in readout.parameters())
    after, after_arrays = evaluate_full(model, corpus, windows, **kwargs)
    assert before["tokens"] == after["tokens"] == 39
    np.testing.assert_array_equal(before_arrays["target_index"], after_arrays["target_index"])
    assert np.any(before_arrays["nll"] != after_arrays["nll"])


def test_invalid_range(model):
    for first, last in [(0, 2), (2, 1), (1, 4)]:
        with pytest.raises(ValueError):
            validate_range(model, first, last)


@pytest.mark.parametrize("bos_id", [None, 0])
def test_chunked_evaluation_matches_native_full_model(model, bos_id):
    corpus = Corpus(np.arange(35, dtype=np.int64) % VOCAB, np.zeros(35, dtype=np.int32), 1)
    windows = make_windows(35, 12, 6)
    metrics, arrays = evaluate_full(model, corpus, windows, batch_size=3, bos_id=bos_id,
                                    device="cpu", chunk=5, progress=False)
    expected_nll, expected_predictions, expected_indices = [], [], []
    with torch.no_grad():
        for batch in iterate_batches(corpus.ids, windows, 3, bos_id):
            logits = model(input_ids=torch.from_numpy(batch.inputs), use_cache=False).logits
            selected = logits[torch.from_numpy(batch.rows), torch.from_numpy(batch.positions)].float()
            targets = torch.from_numpy(corpus.ids[batch.targets])
            expected_nll.append(torch.nn.functional.cross_entropy(selected, targets, reduction="none").numpy())
            expected_predictions.append(selected.argmax(-1).numpy())
            expected_indices.append(batch.targets)
    np.testing.assert_allclose(arrays["nll"], np.concatenate(expected_nll), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(arrays["prediction"], np.concatenate(expected_predictions))
    np.testing.assert_array_equal(arrays["target_index"], np.concatenate(expected_indices))
    assert metrics["cross_entropy"] == pytest.approx(np.concatenate(expected_nll).mean(), rel=1e-6)
    assert len(np.unique(arrays["target_index"])) == metrics["tokens"] == 34


def test_cache_rebatching_preserves_token_reference_alignment(model, tmp_path):
    corpus = Corpus(np.arange(35, dtype=np.int64) % VOCAB, np.zeros(35, dtype=np.int32), 1)
    windows = make_windows(35, 12, 12)
    meta = cache_training(model, corpus, windows, tmp_path, first=2, last=2,
                          batch_size=2, bos_id=None, device="cpu", chunk=5, progress=False, log=lambda _: None)
    dataset = CachedWindows(tmp_path, meta)
    groups = training_batches(dataset, 3, np.random.default_rng(0))
    indices = [index for group in groups for index in group]
    assert sorted(indices) == list(range(len(dataset)))
    segment = make_segment(model, 2, 2)
    count = 0
    for group in groups:
        assert len({dataset.lengths[index] for index in group}) == 1
        batch = collate_windows([dataset[index] for index in group])
        with torch.no_grad():
            actual = segment(inputs_embeds=batch["prefix"], use_cache=False).last_hidden_state
        torch.testing.assert_close(actual[batch["rows"], batch["positions"]], batch["reference_exit"],
                                   atol=1e-5, rtol=1e-5)
        for row, index in enumerate(group):
            item = dataset[index]
            keep = batch["rows"] == row
            torch.testing.assert_close(batch["targets"][keep], item["targets"])
            torch.testing.assert_close(batch["positions"][keep], item["positions"])
        count += len(batch["targets"])
    assert count == meta["tokens"]


def test_bfloat16_segment_preserves_dtype_and_positions(model):
    model.bfloat16()
    states, _ = forward_states(model, torch.randint(0, VOCAB, (1, 12)))
    segment = make_segment(model, 2, 2)
    actual = segment(inputs_embeds=states[1], use_cache=False).last_hidden_state
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, states[2])


@pytest.mark.parametrize("l2_space", ["residual", "rms"])
def test_chunked_backward_matches_unchunked(model, tmp_path, l2_space):
    corpus = Corpus(np.arange(33, dtype=np.int64) % VOCAB, np.zeros(33, dtype=np.int32), 1)
    meta = cache_training(model, corpus, make_windows(33, 12, 12), tmp_path, first=2, last=2,
                          batch_size=1, bos_id=None, device="cpu", chunk=5, log=lambda _: None)
    other = copy.deepcopy(model)
    for current, chunk in [(model, 3), (other, 1000)]:
        train_segment(make_segment(current, 2, 2), Readout(current), tmp_path, meta,
                      device="cpu", epochs=2, lr=1e-4, strength=0.1, regularizer="l2",
                      chunk=chunk, seed=2, log=lambda _: None, l2_space=l2_space)
    for left, right in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_finetune_cli_end_to_end(tmp_path):
    from conftest import tiny
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    from depthprobe.__main__ import main

    model_path = tmp_path / "original"
    original = tiny("gpt2")
    original.save_pretrained(model_path)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(model_path)
    train_file, eval_file = tmp_path / "train.txt", tmp_path / "eval.txt"
    train_file.write_text("a b c " * 12)
    eval_file.write_text("a c b " * 10)
    out = tmp_path / "run"
    main(["finetune-segment", "--model", str(model_path), "--data", str(eval_file),
          "--train-data", str(train_file), "--first-block", "2", "--last-block", "2",
          "--ctx", "16", "--stride", "8", "--chunk", "5", "--batch-size", "2",
          "--device", "cpu", "--inference-device", "cpu", "--out", str(out), "--save-full-model",
          "--l2-space", "rms", "--regularizer", "both", "--reg-strength", "0.5", "--kl-strength", "0.25"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["evaluation"]["before"]["tokens"] == 29
    assert summary["cache"]["first_block"] == summary["cache"]["last_block"] == 2
    assert summary["args"]["l2_space"] == "rms"
    assert summary["args"]["regularizer"] == "both" and summary["args"]["kl_strength"] == 0.25
    restored = AutoModelForCausalLM.from_pretrained(out / "model", local_files_only=True)
    for i in (0, 2):
        for before, after in zip(original.transformer.h[i].parameters(), restored.transformer.h[i].parameters()):
            torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert (out / "segment.pt").exists()
    assert (out / "before.npz").exists() and (out / "after.npz").exists()


def test_parameter_free_rms_formula_and_gradient():
    hidden = torch.tensor([[3.0, 4.0], [0.0, 0.0]], requires_grad=True)
    eps = 1e-6
    actual = parameter_free_rms(hidden, eps)
    expected = hidden / torch.sqrt((hidden ** 2).mean(-1, keepdim=True) + eps)
    torch.testing.assert_close(actual, expected)
    # Unlike LayerNorm, this does not subtract the per-token feature mean.
    assert actual[0].mean() > 0
    actual.sum().backward()
    assert torch.isfinite(hidden.grad).all()
    with pytest.raises(ValueError):
        parameter_free_rms(hidden, 0)


def test_rms_l2_uses_matching_scale_and_backpropagates():
    reference = torch.tensor([[1.0, 2.0, 3.0], [-2.0, 4.0, 1.0], [3.0, 1.0, -2.0]])
    hidden = (reference + torch.tensor([0.2, -0.1, 0.3])).requires_grad_(True)
    scale = parameter_free_rms(reference).std(0).clamp_min(1e-4)
    actual = l2_deviation(hidden, reference, scale, "rms")
    expected = (parameter_free_rms(hidden) - parameter_free_rms(reference)) / scale
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(l2_deviation(reference, reference, scale, "rms"), torch.zeros_like(reference))
    actual.square().mean().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0


@pytest.mark.parametrize("regularizer", ["l2", "kl", "both", "none"])
def test_progress_only_shows_applicable_diagnostics(regularizer):
    row = {"cross_entropy": 3.0, "accuracy": 0.4, "lr": 1e-5, "peak_gpu_gb": 4.0,
           "regularizer": 0.1, "l2_penalty": 0.2, "kl_penalty": 0.3,
           "max_l2_deviation_std": 0.0001 if regularizer in ("l2", "both") else None}
    postfix = training_postfix(row, regularizer)
    assert ("max_dev_std" in postfix) == (regularizer in ("l2", "both"))
    if regularizer in ("l2", "both"):
        assert float(postfix["max_dev_std"]) == 0.0001  # Small real values must not round to 0.00.
    assert ("KL" in postfix) == (regularizer in ("kl", "both"))
    assert ("L2" in postfix) == (regularizer in ("l2", "both"))
    if regularizer == "both":
        assert float(postfix["L2"]) == 0.2 and float(postfix["KL"]) == 0.3


def test_combined_objective_uses_independent_strengths(model, tmp_path):
    rng = np.random.default_rng(4)
    corpus = Corpus(rng.integers(0, VOCAB, 32, dtype=np.int64), np.repeat(np.arange(2), 16).astype(np.int32), 2)
    windows = make_windows(32, 12, 6)
    kwargs = dict(batch_size=2, bos_id=0, device="cpu", chunk=7)
    readout = Readout(model)
    meta = cache_training(model, corpus, windows, tmp_path, first=2, last=2, **kwargs, log=lambda _: None)
    segment = make_segment(model, 2, 2)
    strength, kl_strength = 0.25, 1.5
    history = train_segment(segment, readout, tmp_path, meta, device="cpu", epochs=1, lr=1e-3,
                            strength=strength, kl_strength=kl_strength, regularizer="both", chunk=7,
                            seed=0, l2_space="rms", log=lambda _: None, log_path=tmp_path / "training.csv")
    with (tmp_path / "training.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        expected = (float(row["cross_entropy"]) + strength * float(row["l2_penalty"])
                    + kl_strength * float(row["kl_penalty"]))
        assert float(row["objective"]) == pytest.approx(expected, rel=1e-6)
    assert history[0]["reg_strength"] == strength and history[0]["kl_strength"] == kl_strength
    assert history[0]["l2_penalty"] > 0 and history[0]["kl_penalty"] > 0


@pytest.mark.parametrize("strength,kl_strength", [(0.25, 1.5), (0.0, 1.5), (0.25, 0.0)])
def test_combined_chunked_update_matches_direct_loss(model, strength, kl_strength):
    """Check actual gradients/updates, not just the logged objective arithmetic."""
    other = copy.deepcopy(model)
    readout, direct_readout = Readout(model), Readout(other)
    segment, direct_segment = make_segment(model, 2, 2), make_segment(other, 2, 2)
    width = next(segment.parameters()).shape[-1]
    # Input hidden states and targets are synthetic; reference drift is nonzero.
    prefix = torch.randn(2, 7, width) * 0.1
    rows = torch.arange(2).repeat_interleave(6)
    positions = torch.arange(6).repeat(2)
    with torch.no_grad():
        reference = segment(inputs_embeds=prefix, use_cache=False).last_hidden_state[rows, positions]
        reference = reference + torch.randn_like(reference) * 0.03
    targets = torch.randint(0, VOCAB, (len(rows),))
    scale = parameter_free_rms(reference).std(0).clamp_min(1e-4)
    parameters = [p for p in segment.parameters() if p.requires_grad]
    direct_parameters = [p for p in direct_segment.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    direct_optimizer = torch.optim.SGD(direct_parameters, lr=0.01)
    batch = dict(prefix=prefix, rows=rows, positions=positions, reference_exit=reference, targets=targets)
    measured = segment_step(segment, readout, batch, optimizer, parameters, scale, device="cpu", amp="none",
                            chunk=5, regularizer="both", strength=strength, kl_strength=kl_strength, l2_space="rms")

    hidden = direct_segment(inputs_embeds=prefix, use_cache=False).last_hidden_state[rows, positions]
    logits = direct_readout(hidden).float()
    ce = torch.nn.functional.cross_entropy(logits, targets)
    l2 = ((parameter_free_rms(hidden) - parameter_free_rms(reference)) / scale).square().mean()
    with torch.no_grad():
        teacher_logp = direct_readout(reference).float().log_softmax(-1)
    kl = (teacher_logp.exp() * (teacher_logp - logits.log_softmax(-1))).sum(-1).mean()
    (ce + strength * l2 + kl_strength * kl).backward()
    torch.nn.utils.clip_grad_norm_(direct_parameters, 1.0, error_if_nonfinite=True)
    direct_optimizer.step()
    assert measured["l2"] / len(rows) == pytest.approx(l2.item(), rel=1e-5)
    assert measured["kl"] / len(rows) == pytest.approx(kl.item(), abs=1e-6)
    for actual, expected in zip(parameters, direct_parameters):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
