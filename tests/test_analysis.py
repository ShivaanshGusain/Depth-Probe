import json

import numpy as np

from depthprobe.analysis import DocBootstrap, per_token_e3, plot, summarize, write_csv


def test_bootstrap_point_is_token_weighted_mean():
    values = np.array([1.0, 2.0, 3.0, 10.0])
    doc = np.array([0, 0, 1, 2])
    point, low, high = DocBootstrap(doc, 200, 0).mean(values)
    assert point == values.mean()
    assert low <= point <= high


def test_per_token_monotonicity():
    # Three tokens over four states: monotone, one reversal, correct-then-wrong.
    nll = np.array([[3.0, 3.0, 2.0], [2.0, 1.0, 1.0], [1.0, 2.0, 0.5], [0.5, 0.5, 0.4]])
    kl = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
    top1 = np.array([[False, False, True], [True, False, True], [True, True, False], [True, True, False]])
    flags, depth, settled = per_token_e3({"raw_nll": nll, "raw_kl": kl, "raw_top1": top1}, "raw", 0, 1e-3)
    assert flags["monotone_nll"].tolist() == [True, False, True]
    assert flags["nll_reversals"].tolist() == [0, 1, 0]
    assert flags["monotone_kl"].all()
    assert flags["correct_then_wrong"].tolist() == [False, False, True]
    assert settled.tolist() == [True, True, False]
    assert depth[:2].tolist() == [1, 2]


def test_summarize_and_plot(tmp_path):
    rng = np.random.default_rng(0)
    states, tokens = 4, 50
    arrays = {"context": np.arange(1, tokens + 1), "doc": np.repeat(np.arange(5), 10),
              "native_nll": rng.random(tokens).astype(np.float32)}
    for lens in ("raw", "tuned"):
        arrays[f"{lens}_nll"] = rng.random((states, tokens)).astype(np.float32)
        arrays[f"{lens}_kl"] = rng.random((states, tokens)).astype(np.float32)
        arrays[f"{lens}_entropy"] = rng.random((states, tokens)).astype(np.float32)
        arrays[f"{lens}_top1"] = rng.random((states, tokens)) > 0.5
        arrays[f"{lens}_agree"] = rng.random((states, tokens)) > 0.5
    for key in ("cos", "rel_dist", "norm"):
        arrays[key] = rng.random((states, tokens)).astype(np.float32) + 0.1
    run = tmp_path / "run"
    run.mkdir()
    np.savez(run / "metrics.npz", **arrays)
    (run / "manifest.json").write_text(json.dumps({"args": {"label": "toy"}, "variant": {}}))
    summary = summarize(run, from_layer=1, min_context=11, eps=1e-3, n_boot=50, seed=0)
    assert summary["tokens"] == 40 and summary["documents"] == 4
    assert len(summary["curves"]["raw_nll"][0]) == states
    assert len(summary["curves"]["tuned_gain_kl"][0]) == states - 1
    json.dumps(summary)
    write_csv(tmp_path / "curves.csv", [summary])
    plot(tmp_path, [summary])
    assert (tmp_path / "nll.png").exists() and (tmp_path / "e3_monotonicity.png").exists()
