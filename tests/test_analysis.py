import json

import numpy as np

import pytest

from depthprobe.analysis import (DocBootstrap, isotonic_nonincreasing, per_token_e3, per_token_isotonic, plot,
                                 summarize, write_csv)


def pava_nonincreasing(values):
    """Reference pool-adjacent-violators for one sequence."""
    blocks = []  # [mean, count]
    for value in values:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            mean, count = blocks.pop()
            previous = blocks[-1]
            previous[0] = (previous[0] * previous[1] + mean * count) / (previous[1] + count)
            previous[1] += count
    return np.concatenate([[mean] * count for mean, count in blocks])


def test_isotonic_fit_example():
    fit = isotonic_nonincreasing(np.array([[5.0], [4.0], [4.5], [3.0], [2.0]]))
    np.testing.assert_allclose(fit[:, 0], [5.0, 4.25, 4.25, 3.0, 2.0])


@pytest.mark.parametrize("steps", [1, 2, 5, 13, 25])
def test_isotonic_fit_matches_reference(steps):
    values = np.random.default_rng(steps).normal(size=(steps, 300)).cumsum(0)
    fit = isotonic_nonincreasing(values, chunk=64)
    for column in range(values.shape[1]):
        np.testing.assert_allclose(fit[:, column], pava_nonincreasing(values[:, column]), atol=1e-10)


def test_isotonic_scores():
    # One row per token, transposed to [states, tokens]: monotone, rising, down then up, flat.
    paths = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [1.0, 1.0, 1.0]]).T
    scores, flat = per_token_isotonic(paths, flat_tol=1e-3)
    assert flat.tolist() == [False, False, False, True]
    # [3, 1, 2] fits as [3, 1.5, 1.5]: residual 0.5 of total 2.
    np.testing.assert_allclose(scores["r2"][:3], [1.0, 0.0, 0.75])
    np.testing.assert_allclose(scores["backtrack"][:3], [0.0, 1.0, 1 / 3])
    assert np.isnan(scores["r2"][3]) and np.isnan(scores["backtrack"][3])


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
    for lens in ("raw", "tuned"):
        for metric in ("nll", "kl"):
            entry = summary["isotonic"][lens][metric]
            assert 0 <= entry["r2_mean"][0] <= 1 and 0 <= entry["backtrack_mean"][0] <= 1
            assert set(entry["r2_quantiles"]) == {"0.05", "0.25", "0.5", "0.75", "0.95"}
    json.dumps(summary)
    write_csv(tmp_path / "curves.csv", [summary])
    plot(tmp_path, [summary])
    assert (tmp_path / "nll.png").exists() and (tmp_path / "e3_monotonicity.png").exists()
    assert (tmp_path / "isotonic_r2.png").exists() and (tmp_path / "isotonic_backtrack.png").exists()
