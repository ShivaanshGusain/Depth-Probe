import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from conftest import MODEL_CLASSES, tiny_config
from depthprobe.__main__ import main


def save_tiny_moe(path):
    MODEL_CLASSES["qwen3_moe"](tiny_config("qwen3_moe")).save_pretrained(path)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]").save_pretrained(path)


def test_h1_commands_on_bfloat16_moe(tmp_path):
    model = tmp_path / "moe"
    save_tiny_moe(model)
    data = tmp_path / "text.txt"
    data.write_text("a b c a c b " * 20)
    common = ["--model", str(model), "--dtype", "bfloat16", "--readout-dtype", "model", "--device", "cpu",
              "--data", str(data), "--ctx", "16", "--batch-size", "2", "--chunk", "8"]

    main(["check", *common, "--stride", "8", "--out", str(tmp_path / "check")])
    check = json.loads((tmp_path / "check" / "check.json").read_text())
    assert check["readout_dtype"] == "bfloat16"
    assert check["result"]["max_abs_library_loss_gap"] < 1e-4
    assert check["result"]["max_abs_final_nll_error"] < 0.05

    for name, control in (("pretrained", []), ("random", ["--init", "random", "--init-seed", "0"])):
        main(["train-lens", *common, *control, "--tf32", "--train-tokens", "100", "--val-tokens", "50",
              "--out", str(tmp_path / name / "lens")])
        main(["measure", *common, *control, "--stride", "8", "--lens", str(tmp_path / name / "lens" / "lens.pt"),
              "--out", str(tmp_path / name / "measure")])
        manifest = json.loads((tmp_path / name / "measure" / "manifest.json").read_text())
        assert manifest["readout_dtype"] == "bfloat16"
        assert manifest["stats"]["max_abs_final_nll_error"] < 0.05

    lens = json.loads((tmp_path / "pretrained" / "lens" / "lens.json").read_text())
    assert lens["lens"]["tf32_during_fit"] is True and lens["lens"]["readout_dtype"] == "bfloat16"
    # TF32 is limited to fitting: the lens's validation and everything after it run strict.
    assert lens["environment"]["tf32_matmul"] is False

    # A lens fitted through one readout precision is refused by a measurement using another.
    float32_readout = [arg if arg != "model" else "float32" for arg in common]
    with pytest.raises(SystemExit, match="Lens was trained for") as refusal:
        main(["measure", *float32_readout, "--stride", "8", "--lens", str(tmp_path / "pretrained" / "lens" / "lens.pt"),
              "--out", str(tmp_path / "mismatch")])
    assert "readout_dtype" in str(refusal.value)


def test_lens_without_precision_fields_counts_as_float32_readout(tmp_path):
    import torch

    from depthprobe.lens import load_lens, save_lens

    model = tmp_path / "moe"
    save_tiny_moe(model)
    data = tmp_path / "text.txt"
    data.write_text("a b c a c b " * 20)
    common = ["--model", str(model), "--device", "cpu", "--data", str(data), "--ctx", "16", "--batch-size", "2",
              "--chunk", "8"]
    main(["train-lens", *common, "--train-tokens", "100", "--val-tokens", "50", "--out", str(tmp_path / "lens")])
    # Rewrite the lens as an older version would have saved it: no dtype or readout_dtype in its meta.
    lens, meta = load_lens(tmp_path / "lens" / "lens.pt", "cpu")
    legacy = {key: value for key, value in meta.items() if key not in ("dtype", "readout_dtype")}
    save_lens(tmp_path / "legacy.pt", lens, legacy)
    main(["measure", *common, "--stride", "8", "--lens", str(tmp_path / "legacy.pt"), "--out", str(tmp_path / "ok")])
    assert (tmp_path / "ok" / "manifest.json").exists()
    with pytest.raises(SystemExit, match="Lens was trained for"):
        main(["measure", *common, "--dtype", "bfloat16", "--readout-dtype", "model", "--stride", "8",
              "--lens", str(tmp_path / "legacy.pt"), "--out", str(tmp_path / "refused")])
    assert torch.get_default_dtype() == torch.float32


def test_finetune_refuses_bfloat16_readout_before_any_work(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(SystemExit, match="readout-dtype float32"):
        main(["finetune-segment", "--model", str(tmp_path / "absent"), "--data", str(tmp_path / "absent-data"),
              "--first-block", "1", "--last-block", "1", "--dtype", "bfloat16", "--readout-dtype", "model",
              "--out", str(out)])
    assert not out.exists()
    main(["analyze", str(tmp_path / "pretrained" / "measure"), str(tmp_path / "random" / "measure"),
          "--bootstrap", "20", "--out", str(tmp_path / "report")])
    assert (tmp_path / "report" / "summary.json").exists()
