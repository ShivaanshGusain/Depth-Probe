import json

import numpy as np
import pytest
import torch

from conftest import VOCAB, save_tiny_with_tokenizer, tiny
from depthprobe.__main__ import main
from depthprobe.absorption import (MARGIN, control_verdict, perturbation, q1_verdict, q3_verdict, q4_verdict,
                                   relative_error, rotation, run_absorption, summarize, summarize_tests, swap)
from depthprobe.data import Corpus, make_windows
from depthprobe.model import Readout


def corpus(n=160):
    rng = np.random.default_rng(0)
    return Corpus(rng.integers(0, VOCAB, n).astype(np.int64), np.repeat(np.arange(4), n // 4).astype(np.int32), 4)


def test_noise_follows_the_papers_frobenius_rule_or_per_token_scaling():
    base = torch.randn(2, 6, 8, generator=torch.Generator().manual_seed(0)) * 3
    base[0, 2] *= 50  # a sink-like token with a huge norm
    xi = torch.randn(2, 6, 8, generator=torch.Generator().manual_seed(1))
    keep = torch.ones(2, 6)
    keep[1, 0] = 0
    whole = perturbation(base, xi, 0.01, "frobenius", keep)
    expected = 0.01 * (base * keep[..., None]).flatten(1).norm(dim=-1)
    torch.testing.assert_close(whole.flatten(1).norm(dim=-1), expected)
    per_token = perturbation(base, xi, 0.01, "token", keep)
    torch.testing.assert_close(per_token.norm(dim=-1), 0.01 * base.norm(dim=-1) * keep)
    assert torch.all(whole[1, 0] == 0) and torch.all(per_token[1, 0] == 0)
    # Whole-tensor scaling hands ordinary tokens far more than eta of their own norm when one token is huge.
    ordinary = whole[0, 0].norm() / base[0, 0].norm()
    assert ordinary > 0.05


def test_relative_error_matches_its_definition():
    reference = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(2))
    perturbed = reference * 1.1
    error, distance = relative_error(perturbed, reference, torch.ones(2, 5))
    torch.testing.assert_close(error, torch.full((2,), 0.1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(distance, torch.zeros(2), rtol=0, atol=1e-6)


def test_run_absorption_realises_eta_at_the_site_and_summarizes():
    model = tiny("gpt2")
    text = corpus()
    windows = make_windows(len(text.ids), 32, 32)
    arrays, meta = run_absorption(model, text, windows, sites=[0, 1, 2], etas=[0.0, 1e-2, 1e-1], seeds=2,
                                  batch_size=2, bos_id=None, device="cpu", replicas=3, log=lambda _: None)
    error = arrays["relative_error"]
    assert error.shape == (3, 3, 2, 4, len(windows))
    assert arrays["relative_error_final"].shape == (3, 3, 2, len(windows))
    assert meta["floor_max"] < 1e-5
    for i, site in enumerate(meta["sites"]):
        assert np.isnan(error[i, :, :, :site]).all()  # nothing before the injection site
        for e, eta in enumerate(meta["etas"]):  # float32: realised = intended
            np.testing.assert_allclose(error[i, e, :, site], eta, rtol=1e-4, atol=1e-7)
    rows = summarize(arrays, meta, n_boot=50)
    assert len(rows) == 9
    noisy = [row for row in rows if row["eta"] > 0]
    assert all(row["absorption"] is not None and np.isfinite(row["absorption"]) for row in noisy)
    assert all(row["absorption_low"] <= row["absorption_high"] for row in noisy)
    assert all(row["absorption_final"] is not None and row["nonfinite_fraction"] == 0 for row in noisy)
    assert all(abs(row["realised_over_eta"] - 1) < 1e-3 for row in noisy)
    for row in rows:
        if row["eta"] == 0:  # nothing injected: no ratio, and the comparison itself is (nearly) exact
            assert row["absorption"] is None and row["R_last"] < 1e-5


def test_float16_rounds_a_tiny_perturbation_away():
    model = tiny("llama").to(torch.float16)
    text = corpus()
    arrays, meta = run_absorption(model, text, make_windows(len(text.ids), 32, 32), sites=[1], etas=[1e-7], seeds=1,
                                  batch_size=2, bos_id=None, device="cpu", scale="token", log=lambda _: None)
    rows = summarize(arrays, meta, n_boot=10)
    assert rows[0]["rounded_away_fraction"] > 0.5  # below half an FP16 step the perturbation is never injected


def test_summarize_counts_overflow_instead_of_averaging_it():
    error = np.full((1, 1, 1, 3, 4), np.nan)
    error[0, 0, 0, 1] = [0.1, 0.1, 0.0, 0.1]  # window 2 rounded away
    error[0, 0, 0, 2] = [0.05, np.inf, 0.05, 0.2]  # window 1 overflowed
    arrays = {"relative_error": error, "relative_error_final": error[:, :, :, 2]}
    meta = {"sites": [1], "etas": [0.1], "n_states": 3}
    row = summarize(arrays, meta, n_boot=20)[0]
    assert row["nonfinite_fraction"] == 0.25 and row["rounded_away_fraction"] == 0.25
    assert row["absorption"] == np.mean([0.5, 2.0]) and row["absorption_median"] == np.median([0.5, 2.0])
    assert np.isfinite(row["R_last"])


def test_reproduce_absorption_cli(tmp_path):
    model = tmp_path / "model"
    save_tiny_with_tokenizer(model, "gpt2")
    data = tmp_path / "text.txt"
    data.write_text("".join(f" = T{i} = \n" + "a b c a c b " * 8 + "\n" for i in range(6)))
    out = tmp_path / "run"
    main(["reproduce-absorption", "--model", str(model), "--device", "cpu", "--data", str(data), "--ctx", "16",
          "--stride", "16", "--batch-size", "2", "--sites", "0,2", "--etas", "1e-2,1e-1", "--seeds", "2",
          "--bootstrap", "20", "--out", str(out)])
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["summary"]) == 4 and manifest["absorption"]["scale"] == "frobenius"
    assert (out / "absorption.png").exists() and (out / "absorption.npz").exists()


def test_swap_is_keyed_and_replaces_one_early_token():
    ids = np.random.default_rng(3).integers(0, VOCAB, 500)
    row = ids[:32]
    position, token = swap(0, 100, row, ids, bos_offset=0)
    assert (position, token) == swap(0, 100, row, ids, bos_offset=0)  # same window and seed: same swap
    assert 1 <= position <= 30 and token != row[position]
    position, _ = swap(0, 100, row, ids, bos_offset=1)
    assert position >= 2  # never BOS or the first token


def test_absolute_sizes_and_real_swaps_on_a_tiny_model():
    model = tiny("llama")
    text = corpus()
    windows = make_windows(len(text.ids), 32, 32)
    with pytest.raises(ValueError, match="sites >= 1"):
        run_absorption(model, text, windows, sites=[0], etas=[1e-2], seeds=1, batch_size=2, bos_id=None,
                       device="cpu", readout=Readout(model), swaps=True, log=lambda _: None)
    arrays, meta = run_absorption(model, text, windows, sites=[1, 2], etas=[1e-2], seeds=2, batch_size=2,
                                  bos_id=None, device="cpu", scale="token", replicas=3, readout=Readout(model),
                                  swaps=True, log=lambda _: None)
    assert arrays["delta_norm"].shape == (3, 4, 2, 1, 2, 4, len(windows))
    noise, real, rotated = arrays["delta_norm"]
    clean = arrays["clean_norm"]  # [site, mask, seed, state, window]
    for i, site in enumerate(meta["sites"]):
        # Per-token scaling: on any token set the injected size is exactly eta times the clean size.
        np.testing.assert_allclose(noise[0, i, 0, :, site] / clean[i, 0, :, site], 1e-2, rtol=1e-4)
        np.testing.assert_allclose(real[1, i, 0, :, site] / clean[i, 1, :, site], 1e-2, rtol=1e-4)
        np.testing.assert_allclose(rotated[1, i, 0, :, site] / clean[i, 1, :, site], 1e-2, rtol=1e-4)
        # The absolute sizes reproduce the relative error measured the old way.
        np.testing.assert_allclose(noise[0, i, 0, :, 3] / clean[i, 0, :, 3], arrays["relative_error"][i, 0, :, 3],
                                   rtol=1e-4)
    assert np.isnan(real[0]).all() and np.isnan(rotated[0]).all()  # they live on the swapped tokens only
    assert arrays["changed_tokens"].shape == (2, len(windows), 32) and arrays["changed_tokens"].dtype == bool
    counts = arrays["counts"]
    np.testing.assert_array_equal(counts[:, 1], counts[:, 2] + counts[:, 3])  # swapped = changed + preserved
    assert counts[:, 1].sum() > 0
    assert np.nanmin(arrays["kl"][0, 0]) >= -1e-6 and np.nanmin(arrays["kl_carry"][0, 0]) >= -1e-6
    assert np.all((arrays["swap_positions"] >= 1) & (arrays["swap_positions"] < 31))
    # Massive coordinates only ever remove energy; the random direction at the last state has size eta exactly.
    full, bulk = arrays["delta_norm"], arrays["delta_norm_bulk"]
    measured = np.isfinite(full)
    assert np.array_equal(measured, np.isfinite(bulk)) and np.all(bulk[measured] <= full[measured] * (1 + 1e-9))
    np.testing.assert_allclose(arrays["size_random_total"][:, 0], 1e-4 * counts[:, 0], rtol=1e-12)
    for key in ("kl_total", "kl_random_total"):
        assert np.nanmin(arrays[key]) >= -1e-12, key  # float64 KL: no negative rounding
    assert np.nanmin(arrays["size_total"]) >= 0 and np.nanmin(arrays["size_post_total"]) >= 0
    # The Q4 reference has exactly the propagated perturbation's size at every token.
    np.testing.assert_allclose(arrays["size_total"][:, 2], arrays["size_total"][:, 0], rtol=0)
    rows = summarize_tests(arrays, meta, n_boot=50)
    assert len(rows) == 2 and all("q1" in row and "q3" in row and "output_kl_real" in row for row in rows)
    assert all(np.isfinite(row["log_repaired_noise"]["value"]) for row in rows)
    for row in rows:
        assert row["q3_statistic"] == "real vs rotated real" and "q3_vs_noise" in row and "q3_massive_check" in row
        assert np.isfinite(row["log_rotated_over_noise_changed"]["value"]) or row["changed_fraction"] == 0
        assert "q4" in row and np.isfinite(row["log_steering_noise"]["value"])
        assert np.isfinite(row["log_steering_noise_post_norm"]["value"])
        assert np.isfinite(row["log_steering_real"]["value"]) and np.isfinite(row["log_steering_rotated"]["value"])
        shares = [row[f"massive_share_{kind}_{where}"] for kind in ("noise", "real", "rotated")
                  for where in ("site", "last")]
        assert all(share is None or -1e-9 <= share <= 1 for share in shares)


def test_rotation_keeps_sizes_and_token_correlations():
    q = rotation(0, 16)
    torch.testing.assert_close(q @ q.T, torch.eye(16), atol=1e-5, rtol=0)
    difference = torch.randn(5, 16, generator=torch.Generator().manual_seed(4))
    turned = difference @ q
    torch.testing.assert_close(turned.norm(dim=-1), difference.norm(dim=-1), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(turned @ turned.T, difference @ difference.T, rtol=1e-4, atol=1e-4)
    assert torch.equal(rotation(0, 16), q) and not torch.allclose(rotation(1, 16), q)


def test_control_rules_skip_saturated_and_reported_only_controls():
    rows = [{"control": "shuffled", "site": 1, "eta": 0.01, "x": "yes"},
            {"control": "permuted", "site": 1, "eta": 0.01, "x": "not applicable (control saturated)"},
            {"control": "random", "site": 1, "eta": 0.01, "x": "no"}]
    assert control_verdict(rows, 1, 0.01, "x", deciding=["shuffled", "permuted"]) == "yes"
    assert control_verdict(rows, 1, 0.01, "x") == "no"  # with every control deciding, random's "no" counts
    assert control_verdict(rows[1:2], 1, 0.01, "x", deciding=["permuted"]).startswith("not applicable")
    assert control_verdict(rows, 2, 0.01, "x") == "not measured"


def interval(value, half=0.05):
    return {"value": value, "low": value - half, "high": value + half}


def test_verdict_rules():
    assert q1_verdict(interval(np.log(0.3))) == "shrinks"
    assert q1_verdict(interval(0.0)) == "carried unchanged (dilution)"
    assert q1_verdict(interval(-0.2, 0.2)) == "unclear"
    assert q3_verdict(interval(1.0), interval(1.0)) == "denoising"
    assert q3_verdict(interval(1.0), interval(0.0)) == "convergence"
    assert q3_verdict(interval(0.0), interval(0.0)) == "compression"
    assert q3_verdict(interval(1.0), interval(0.2, 0.2)) == "unclear"
    assert q3_verdict(interval(1.0), interval(-1.0)).startswith("none of the three")
    assert q4_verdict(interval(np.log(0.4))).startswith("steered away")
    assert q4_verdict(interval(0.0)).startswith("not steered")
    assert q4_verdict(interval(np.log(1.7))).startswith("steered toward")
    assert q4_verdict(interval(0.2, 0.2)) == "unclear"
    assert abs(MARGIN - np.log(1.25)) < 1e-12


def test_absorption_tests_cli(tmp_path):
    model = tmp_path / "model"
    save_tiny_with_tokenizer(model, "llama")
    data = tmp_path / "text.txt"
    data.write_text("".join(f" = T{i} = \n" + "a b c a c b " * 8 + "\n" for i in range(6)))
    common = ["--model", str(model), "--device", "cpu", "--data", str(data), "--ctx", "16", "--stride", "16",
              "--batch-size", "2", "--sites", "1,2", "--etas", "1e-3,1e-2", "--seeds", "2", "--scale", "token",
              "--exclude-sinks", "--bootstrap", "20"]
    trained = tmp_path / "trained"
    main(["reproduce-absorption", *common, "--real-swaps", "--label", "trained", "--out", str(trained)])
    for label, init in (("random", "random"), ("permuted", "permuted")):
        main(["reproduce-absorption", *common, "--init", init, "--real-swaps", "--sinks-from", str(trained),
              "--label", label, "--out", str(tmp_path / label)])
    with np.load(trained / "absorption.npz") as first, np.load(tmp_path / "random" / "absorption.npz") as second:
        np.testing.assert_array_equal(first["changed_tokens"], second["changed_tokens"])  # the trained token sets
    main(["reproduce-absorption", *common, "--init", "permuted", "--label", "own-sinks",
          "--out", str(tmp_path / "own-sinks")])
    main(["analyze-absorption", str(trained), str(tmp_path / "random"), str(tmp_path / "permuted"), "--eta", "1e-3",
          "--bootstrap", "50", "--out", str(tmp_path / "report")])
    verdict = json.loads((tmp_path / "report" / "verdict.json").read_text())
    assert verdict["deciding_controls"] == ["permuted"]  # random weights are reported only
    assert [answer["site"] for answer in verdict["answers"]] == [1, 2]
    for answer in verdict["answers"]:
        for key in ("q2_learned", "q2_relative_learned", "q3_learned", "q4_trained_steers_away_more"):
            result = answer[key]["answer"]
            assert result in ("yes", "no", "unclear") or result.startswith("not applicable"), (key, result)
            assert "random: " in answer[key]["per_control"] and "(reported only)" in answer[key]["per_control"]
        assert not answer["q4_noise_steering"].startswith("not measured")
        assert not answer["q3_without_massive_coordinates"].startswith("not measured")
        assert answer["q3_kind"] in ("denoising", "convergence", "compression", "unclear") \
            or answer["q3_kind"].startswith("none of the three")
    assert (tmp_path / "report" / "tests.png").exists()
    with pytest.raises(SystemExit, match="reuse the reference sinks"):
        main(["analyze-absorption", str(trained), str(tmp_path / "own-sinks"), "--out", str(tmp_path / "refused")])
    with pytest.raises(SystemExit, match="different"):
        main(["reproduce-absorption", *[a if a != "16" else "8" for a in common], "--sinks-from", str(trained),
              "--out", str(tmp_path / "mismatch")])
