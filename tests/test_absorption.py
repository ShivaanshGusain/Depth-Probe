import json

import numpy as np
import torch

from conftest import VOCAB, save_tiny_with_tokenizer, tiny
from depthprobe.__main__ import main
from depthprobe.absorption import perturbation, relative_error, run_absorption, summarize
from depthprobe.data import Corpus, make_windows


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
