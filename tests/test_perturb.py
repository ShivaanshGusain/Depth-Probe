import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import H1_KINDS, VOCAB, save_tiny_with_tokenizer, tiny
from depthprobe.__main__ import main
from depthprobe.data import Corpus, make_windows
from depthprobe.finetune import make_segment
from depthprobe.model import Readout, ResidualCapture, find_parts, load_model, n_blocks, shuffle_permutation
from depthprobe.perturb import (FAMILIES, calibrate, check_layer_types, propagation_metrics, run_perturbation,
                                shared_direction, sink_mask, unit_directions, window_noise)
from depthprobe.perturb_report import gate
from depthprobe.toys import toy_signatures


def ids(batch=2, length=16):
    return torch.randint(0, VOCAB, (batch, length), generator=torch.Generator().manual_seed(1))


def corpus(n=160):
    rng = np.random.default_rng(0)
    return Corpus(rng.integers(0, VOCAB, n).astype(np.int64), np.repeat(np.arange(4), n // 4).astype(np.int32), 4)


@pytest.mark.parametrize("kind", H1_KINDS)
def test_zero_noise_suffix_segment_reproduces_the_full_model(kind):
    model = tiny(kind)
    tokens = ids()
    with ResidualCapture(model) as capture:
        find_parts(model).container(input_ids=tokens, use_cache=False)
    clean = capture.states
    logits = model(input_ids=tokens, use_cache=False).logits.float()
    blocks_count = n_blocks(model)
    for site in range(blocks_count):
        segment = make_segment(model, site + 1, blocks_count, trainable=False)
        blocks = getattr(segment, find_parts(model).blocks_attr)
        with ResidualCapture(blocks=blocks) as captured:
            output = segment(inputs_embeds=torch.cat([clean[site], clean[site]]), use_cache=False)
        assert len(captured.states) == blocks_count - site + 1
        for offset, state in enumerate(captured.states):
            for half in (state[:2], state[2:]):  # both copies in the doubled batch
                torch.testing.assert_close(half, clean[site + offset], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(Readout(model)(output.last_hidden_state[:2]), logits, rtol=1e-4, atol=1e-4)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_capture_can_keep_only_selected_tokens():
    model = tiny("llama")
    tokens = ids()
    rows, positions = torch.tensor([0, 1, 1]), torch.tensor([3, 0, 7])
    with ResidualCapture(model) as full:
        find_parts(model).container(input_ids=tokens, use_cache=False)
    with ResidualCapture(model, select=(rows, positions)) as selected:
        find_parts(model).container(input_ids=tokens, use_cache=False)
    for whole, part in zip(full.states, selected.states):
        torch.testing.assert_close(part, whole[rows, positions])


def test_noise_is_unit_sized_masked_and_keyed_by_window():
    width, length = 16, 10
    z, signs = window_noise(0, [5, 40], length, width)
    alone, _ = window_noise(0, [40], length, width)
    torch.testing.assert_close(z[1], alone[0])  # the same window gets the same draws in any batch
    bulk = torch.ones(width)
    bulk[3] = 0
    shared = shared_direction(0, width)
    iso = unit_directions("iso", z, shared, signs, None, bulk)
    torch.testing.assert_close(iso.norm(dim=-1), torch.ones(2, length))
    assert torch.all(iso[..., 3] == 0)
    same, flipped = (unit_directions(family, z, shared, signs, None, bulk) for family in ("shared", "signed"))
    torch.testing.assert_close(flipped.abs(), same.abs())
    torch.testing.assert_close(same[0, 0], same[1, 9])  # one direction for every token
    raw = torch.randn(width, 4, generator=torch.Generator().manual_seed(2))
    raw[3] = 0  # zero the massive row before orthonormalizing, as calibration's embedded bases are
    basis = torch.linalg.qr(raw)[0]
    projected = unit_directions("on", z, shared, signs, basis, bulk)
    torch.testing.assert_close(projected, (projected @ basis) @ basis.T, rtol=1e-5, atol=1e-5)
    assert torch.all(projected[..., 3].abs() < 1e-7)


def test_calibration_bases_are_orthonormal_disjoint_and_skip_massive_coordinates():
    model = tiny("llama")
    with torch.no_grad():
        model.model.embed_tokens.weight[:, 5] += 50.0  # a massive coordinate the residual path carries everywhere
    text = corpus(200)
    calibration = calibrate(model, Readout(model), text, make_windows(len(text.ids), 32, 16), batch_size=2,
                            bos_id=None, device="cpu", log=lambda _: None)
    for state in range(n_blocks(model) + 1):
        assert 5 in calibration["massive"][state]
        bulk = calibration["bulk"][state]
        for key in ("on", "off", "read", "unread"):
            basis = calibration[key][state]
            torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1]), rtol=0, atol=1e-5)
            assert torch.all(basis[bulk == 0] == 0) and torch.all(basis[5] == 0)
        assert (calibration["on"][state].T @ calibration["off"][state]).abs().max() < 1e-5
        assert (calibration["read"][state].T @ calibration["unread"][state]).abs().max() < 1e-5
        assert 0 < calibration["on_share"][state] <= 1 and calibration["on_rank"][state] >= 1


def test_sink_mask_marks_first_position_and_activation_outliers():
    states = [torch.ones(1, 6, 4) for _ in range(3)]
    states[2][0, 3] = 100.0
    assert sink_mask(states, 0, 10.0).tolist() == [[True, False, False, True, False, False]]
    assert sink_mask(states, 1, 10.0)[0, 1]  # a prepended BOS is a sink too


def test_mixed_attention_layer_types_are_rejected():
    check_layer_types(SimpleNamespace(layer_types=["full_attention"] * 3))
    check_layer_types(SimpleNamespace())
    with pytest.raises(ValueError):
        check_layer_types(SimpleNamespace(layer_types=["full_attention", "sliding_attention"]))


def test_propagation_metrics_on_a_pure_contraction():
    clean = [torch.randn(5, 8, generator=torch.Generator().manual_seed(0)) * 5.0 * 0.5 ** k for k in range(4)]
    eps = torch.randn(5, 8, generator=torch.Generator().manual_seed(1)) * 0.1
    noisy = [clean[k] + eps * 0.5 ** k for k in range(4)]
    metrics = propagation_metrics(clean, noisy, "rms")
    expected = torch.tensor([k * math.log(0.5) for k in range(4)])[:, None].expand(4, 5)
    torch.testing.assert_close(metrics["log_survival"], expected, rtol=0, atol=1e-5)
    torch.testing.assert_close(metrics["log_A"], torch.zeros(4, 5), rtol=0, atol=1e-4)  # scaling is invisible
    torch.testing.assert_close(metrics["retained"], torch.tensor([0.5 ** k for k in range(4)])[:, None].expand(4, 5),
                               rtol=0, atol=1e-5)


def test_subspace_survival_is_undefined_without_an_injected_component():
    basis = torch.eye(6)
    on, off = [basis[:, :3]] * 2, [basis[:, 3:]] * 2
    clean = [torch.ones(4, 6) * 3.0] * 2
    eps = torch.zeros(4, 6)
    eps[:, 0] = 0.1  # entirely on-manifold
    metrics = propagation_metrics(clean, [clean[0] + eps, clean[1] + 0.5 * eps], "rms", off=off, on=on)
    assert torch.isnan(metrics["log_off"]).all() and torch.isnan(metrics["log_off0"]).all()
    torch.testing.assert_close(metrics["log_on"][1], torch.full((4,), math.log(0.5)), rtol=0, atol=1e-5)
    torch.testing.assert_close(metrics["log_on0"][1], torch.zeros(4), rtol=0, atol=1e-6)  # carried unchanged


def test_toys_have_their_known_signatures():
    results = toy_signatures()
    assert results["contraction"]["iso"]["log_survival"] == pytest.approx(9 * math.log(0.8), abs=1e-4)
    for name in ("linear_diffusion", "mixture_diffusion"):  # off-manifold noise shrinks by sigma_L / sigma_b
        assert results[name]["off"]["log_off"] == pytest.approx(results[name]["expected_off_survival"], abs=1e-3)
    for name in ("linear_diffusion", "mixture_diffusion", "manifold_attractor"):  # denoisers: selective
        for key in ("selectivity", "selectivity_A", "selectivity_Lambda"):
            assert results[name][key] > 1.0, (name, key)
    rotation = results["rotation"]
    assert abs(rotation["iso"]["log_survival"]) < 1e-4
    assert abs(rotation["selectivity"]) < 1e-4 and abs(rotation["selectivity_A"]) < 1e-4
    pump = results["norm_pump"]
    assert abs(pump["iso"]["log_survival"]) < 1e-4 and pump["iso"]["log_A"] < -0.1  # dilution only
    assert abs(pump["iso"]["log_A"] - pump["iso"]["log_A0"]) < 1e-4 and abs(pump["selectivity_Lambda"]) < 1e-4
    for name in ("point_attractor", "rotation"):  # no selective removal
        assert abs(results[name]["selectivity_Lambda"]) < 0.1, name
    assert abs(results["point_attractor"]["selectivity"]) < 0.1


def test_permuted_init_shuffles_only_block_matrices(tmp_path):
    reference = tiny("llama")
    reference.save_pretrained(tmp_path)
    permuted = load_model(tmp_path, init="permuted", init_seed=3)
    again = load_model(tmp_path, init="permuted", init_seed=3)
    changed = 0
    for name, value in reference.state_dict().items():
        new = permuted.state_dict()[name]
        if ".layers." in name and value.ndim >= 2:
            assert torch.equal(value.flatten().sort().values, new.flatten().sort().values), name
            changed += int(not torch.equal(value, new))
        else:
            assert torch.equal(value, new), name
        assert torch.equal(new, again.state_dict()[name])
    assert changed > 0


@pytest.mark.parametrize("kind", ["gpt2", "qwen3", "olmoe"])
def test_run_perturbation_accumulates_consistent_sums(kind):
    model = tiny(kind)
    readout = Readout(model)
    text = corpus()
    windows = make_windows(len(text.ids), 32, 16)
    calibration = calibrate(model, readout, text, windows, batch_size=2, bos_id=None, device="cpu", log=lambda _: None)
    arrays, meta = run_perturbation(model, readout, text, windows, calibration, sites=[0, 2], families=list(FAMILIES),
                                    deltas=[0.01, 0.5], seed=0, batch_size=2, bos_id=None, replicas=4, min_context=4,
                                    sink_factor=1e9, chunk=16, device="cpu", log=lambda _: None)
    conditions, states_count = meta["conditions"], n_blocks(model) + 1
    assert arrays["state_log_A"].shape == (len(conditions), states_count, 4)
    assert len(conditions) == 2 * len(FAMILIES) * 2 * 2
    assert meta["floor_max"] < 1e-4  # in-batch clean replica matches the separate clean pass
    for i, condition in enumerate(conditions):
        assert np.all(arrays["state_log_survival"][i, :condition["site"]] == 0)  # nothing before the site
        np.testing.assert_allclose(arrays["state_log_survival"][i, condition["site"]], 0, atol=1e-4)
        if condition["family"] == "on":  # off-manifold survival of an on-manifold injection is undefined
            assert np.all(arrays["defined_off"][i] == 0) and np.all(arrays["state_log_off"][i] == 0)
        if condition["family"] == "off":
            assert np.all(arrays["defined_on"][i] == 0)
        if condition["family"] == "iso":  # undefined tokens are counted out, never turned into NaN sums
            assert np.all(np.isfinite(arrays["state_log_off"][i])) and arrays["defined_off"][i].sum() > 0
    total = arrays["counts"].sum()
    assert total > 0
    nl = arrays["output_nl"].sum(-1) / total
    small = [i for i, c in enumerate(conditions) if c["delta"] == 0.01]
    assert np.all(nl[small] < 0.2)  # tiny antithetic pairs respond almost linearly
    iso = [i for i, c in enumerate(conditions) if c["family"] == "iso"]
    for key in ("kl", "kl_carry", "kl_last"):
        assert np.all(arrays[f"output_{key}"][iso].sum(-1) > 0), key
    if not any(meta["calibration"]["massive"]):  # without massive coordinates bulk = full norm: size = delta
        effective = arrays["output_delta_eff"].sum(-1) / total
        np.testing.assert_allclose(effective, [c["delta"] for c in conditions], rtol=1e-3)
    # The sink record lets a control reuse exactly these measured tokens.
    assert len(arrays["window_starts"]) == len(windows) and len(arrays["sink_starts"]) >= len(windows)


def test_gate_requires_every_deciding_control_under_the_rule():
    rows = [{"control": control, "statistic": f"selectivity_{definition}", "delta": 0.1, "site": site,
             "value": value, "low": value - 0.1, "high": value + 0.1}
            for control, value in (("permuted", 0.5), ("shuffled", 0.5), ("random", 0.0))
            for definition in ("pca", "readout") for site in range(3)]
    passed = gate(rows, ["permuted", "shuffled"], 0.1, 0.22, 3, "both")
    assert passed["passes"] and not passed["per_control"]["random"]["deciding"]
    assert not gate(rows, ["permuted", "shuffled", "random"], 0.1, 0.22, 3, "both")["passes"]
    assert not gate(rows, ["permuted", "missing"], 0.1, 0.22, 3, "both")["passes"]
    one_definition = [row for row in rows if row["statistic"] == "selectivity_pca"]
    assert not gate(one_definition, ["permuted"], 0.1, 0.22, 3, "both")["passes"]
    assert gate(one_definition, ["permuted"], 0.1, 0.22, 3, "either")["passes"]
    with pytest.raises(ValueError, match="Unknown gate rule"):
        gate(rows, ["permuted"], 0.1, 0.22, 3, "all")


def test_perturb_cli_end_to_end_on_a_bfloat16_moe(tmp_path):
    model = tmp_path / "moe"
    save_tiny_with_tokenizer(model, "qwen3_moe")
    data = tmp_path / "text.txt"  # several WikiText-style articles, so the bootstrap has real documents
    data.write_text("".join(f" = T{i} = \n" + "a b c a c b " * 8 + "\n" for i in range(6)))
    common = ["--model", str(model), "--dtype", "bfloat16", "--device", "cpu", "--data", str(data), "--ctx", "16",
              "--stride", "8", "--batch-size", "2", "--chunk", "8", "--every", "1", "--sites", "0,2",
              "--families", "iso,off,on,read,unread", "--deltas", "0.1,0.5", "--replicas", "4",
              "--min-context", "4", "--cal-tokens", "100"]
    trained = tmp_path / "trained"
    main(["perturb", *common, "--label", "trained", "--out", str(trained)])
    # Three blocks leave one middle block, so a "middle" shuffle would be the identity: shuffle all of them.
    seed = next(s for s in range(100) if shuffle_permutation(3, s, "all") != [0, 1, 2])
    for label, control in (("random", ["--init", "random"]),
                           ("shuffled", ["--shuffle", "all", "--shuffle-seed", str(seed)]),
                           ("permuted", ["--init", "permuted"])):
        main(["perturb", *common, *control, "--sinks-from", str(trained), "--label", label,
              "--out", str(tmp_path / label)])
    main(["perturb", *common, "--calibration", str(trained / "calibration.pt"), "--label", "reused",
          "--out", str(tmp_path / "reused")])
    assert not (tmp_path / "reused" / "calibration.pt").exists()
    with np.load(trained / "perturb.npz") as first, np.load(tmp_path / "reused" / "perturb.npz") as second:
        for key in ("state_log_A", "output_kl", "counts"):
            np.testing.assert_array_equal(first[key], second[key])
    with pytest.raises(SystemExit, match="different model variant"):
        main(["perturb", *common, "--init", "permuted", "--calibration", str(trained / "calibration.pt"),
              "--out", str(tmp_path / "refused")])
    runs = [str(tmp_path / label) for label in ("trained", "random", "shuffled", "permuted")]
    main(["analyze-perturb", *runs, "--reference", "trained", "--delta", "0.1", "--min-sites", "1",
          "--bootstrap", "50", "--out", str(tmp_path / "report")])
    summary = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert summary["reference"] == "trained" and summary["contrasts"]
    deciding = summary["gate"]["deciding"]
    assert "permuted" in deciding and "shuffled" in deciding and "random" not in deciding  # random does not decide
    # The pre-registered middle shuffles were not run, so they are listed as missing and the gate cannot pass.
    assert "missing: shuffled middle seed 0" in deciding and not summary["gate"]["passes"]
    with open(tmp_path / "shuffled" / "manifest.json") as handle:
        assert json.load(handle)["permutation"] != [0, 1, 2]
    assert summary["runs"][0]["perturb_meta"]["documents"] > 1
    assert any(cell["log_A"]["low"] < cell["log_A"]["high"] for cell in summary["runs"][0]["cells"])
    for name in ("selectivity.png", "dose_response.png", "noise_vs_signal.png", "absorption_trained.png",
                 "cells.csv", "contrasts.csv"):
        assert (tmp_path / "report" / name).exists(), name
    with pytest.raises(SystemExit, match="was not measured"):
        main(["analyze-perturb", *runs, "--delta", "0.2", "--out", str(tmp_path / "bad-delta")])
    main(["perturb", *common, "--shuffle", "middle", "--label", "own-sinks", "--out", str(tmp_path / "own")])
    if not np.array_equal(np.load(tmp_path / "own" / "perturb.npz")["counts"], np.load(trained / "perturb.npz")["counts"]):
        with pytest.raises(SystemExit, match="measured tokens"):
            main(["analyze-perturb", str(trained), str(tmp_path / "own"), "--out", str(tmp_path / "bad-tokens")])
    main(["perturb-toys", "--out", str(tmp_path / "toys")])
    assert "linear_diffusion" in json.loads((tmp_path / "toys" / "toys.json").read_text())
