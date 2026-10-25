import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import H1_KINDS, MODEL_CLASSES, VOCAB, tiny, tiny_config
from depthprobe.data import Corpus, make_windows
from depthprobe.lens import TunedLens, evaluate_lens, load_lens, save_lens, train_lens
from depthprobe.measure import check, layer_metrics, measure, unigram_logp
from depthprobe.model import Readout, hidden_width, load_model, n_blocks


@pytest.fixture(params=H1_KINDS)
def model(request):
    return tiny(request.param)


def corpus(n=120):
    rng = np.random.default_rng(0)
    return Corpus(rng.integers(0, VOCAB, n).astype(np.int64), np.repeat(np.arange(4), n // 4).astype(np.int32), 4)


def test_layer_metrics_match_reference_formulas():
    torch.manual_seed(0)
    logits, final, targets = torch.randn(6, 11), torch.randn(6, 11), torch.randint(0, 11, (6,))
    logp_final = torch.log_softmax(final, -1)
    metrics = layer_metrics(logits, logp_final, final.argmax(-1), targets)
    torch.testing.assert_close(metrics["nll"], F.cross_entropy(logits, targets, reduction="none"))
    reference_kl = F.kl_div(torch.log_softmax(logits, -1), logp_final, log_target=True, reduction="none").sum(-1)
    torch.testing.assert_close(metrics["kl"], reference_kl)
    order = logits.argsort(-1, descending=True)
    expected_rank = (order == targets[:, None]).float().argmax(-1)
    assert torch.equal(metrics["rank"], expected_rank)
    same = layer_metrics(final, logp_final, final.argmax(-1), targets)
    torch.testing.assert_close(same["kl"], torch.zeros(6), atol=1e-6, rtol=0)
    assert same["agree"].all()


def test_check_matches_library_loss(model):
    text = corpus()
    result = check(model, Readout(model), text, make_windows(len(text.ids), 32, 16), batch_size=2, bos_id=None,
                   chunk=7, device="cpu", log=lambda _: None)
    assert result["scored_tokens"] == len(text.ids) - 1
    assert result["max_abs_library_loss_gap"] < 1e-4
    assert result["max_abs_final_nll_error"] < 1e-4


def test_check_runs_on_bfloat16_model():
    # Label alignment and the final-state comparison on a bfloat16 model. The exact match of a
    # model-dtype readout is tested in test_model.py; here rows are gathered, so it need not be bitwise.
    model = tiny("llama").to(torch.bfloat16)
    text = corpus()
    result = check(model, Readout(model, dtype=torch.bfloat16), text, make_windows(len(text.ids), 32, 16),
                   batch_size=2, bos_id=None, chunk=7, device="cpu", log=lambda _: None)
    assert result["max_abs_library_loss_gap"] < 1e-4
    assert result["max_abs_final_nll_error"] < 0.05
    assert 0 <= result["mean_abs_final_nll_error"] <= result["max_abs_final_nll_error"]


def test_check_ignores_moe_router_loss(tmp_path):
    config = tiny_config("qwen3_moe", output_router_logits=True, router_aux_loss_coef=1.0)
    MODEL_CLASSES["qwen3_moe"](config).save_pretrained(tmp_path)
    model = load_model(tmp_path)
    text = corpus()
    result = check(model, Readout(model), text, make_windows(len(text.ids), 32, 16), batch_size=2, bos_id=None,
                   chunk=7, device="cpu", log=lambda _: None)
    assert result["max_abs_library_loss_gap"] < 1e-4


def test_measure_shapes_and_final_state_consistency(model):
    text = corpus()
    lens = TunedLens(n_blocks(model) + 1, hidden_width(model))
    unigram = unigram_logp(text.ids, VOCAB)
    arrays, stats = measure(model, Readout(model), text, make_windows(len(text.ids), 32, 16), batch_size=2,
                            bos_id=None, chunk=10, device="cpu", lens=lens, unigram_logp=unigram,
                            log=lambda _: None)
    states, tokens = n_blocks(model) + 1, len(text.ids) - 1
    assert stats["scored_tokens"] == tokens
    assert arrays["raw_nll"].shape == (states, tokens)
    assert sorted(arrays["target_index"].tolist()) == list(range(1, len(text.ids)))
    np.testing.assert_allclose(arrays["raw_nll"][-1], arrays["native_nll"], atol=1e-4)
    np.testing.assert_allclose(arrays["raw_kl"][-1], 0, atol=1e-5)
    np.testing.assert_allclose(arrays["cos"][-1], 1, atol=1e-5)
    # A zero-initialized tuned lens is the identity, so it must equal the raw lens.
    np.testing.assert_allclose(arrays["tuned_nll"], arrays["raw_nll"], atol=1e-5)
    np.testing.assert_allclose(arrays["unigram_nll"], -unigram[arrays["target_id"]], atol=1e-6)


def test_lens_trains_saves_and_loads(model, tmp_path):
    text = corpus(200)
    windows = make_windows(len(text.ids), 32, 32)
    readout = Readout(model)
    lens, history = train_lens(model, readout, text.ids, windows, batch_size=2, bos_id=None, chunk=16, lr=1e-2,
                               warmup_fraction=0.1, device="cpu", log=lambda _: None)
    assert len(history) > 0 and np.isfinite(history).all()
    evaluation = evaluate_lens(model, readout, lens, text.ids, windows, batch_size=2, bos_id=None, chunk=16,
                               device="cpu")
    assert evaluation["tuned_kl"][-1] == evaluation["raw_kl"][-1]
    save_lens(tmp_path / "lens.pt", lens, {"n_states": lens.n_states, "width": hidden_width(model),
                                           "validation": evaluation})
    loaded, meta = load_lens(tmp_path / "lens.pt", "cpu")
    hidden = torch.randn(3, hidden_width(model))
    torch.testing.assert_close(loaded(hidden, 0), lens(hidden, 0))
    assert meta["validation"]["tokens"] == evaluation["tokens"]
