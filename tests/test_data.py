import numpy as np
import pytest

from depthprobe.data import iterate_batches, make_windows, split_documents


@pytest.mark.parametrize("n, ctx, stride", [(50, 16, 16), (50, 16, 8), (17, 16, 4), (5, 16, 8), (64, 16, 1)])
def test_each_target_scored_at_most_once(n, ctx, stride):
    windows = make_windows(n, ctx, stride)
    targets = np.concatenate([np.arange(w.first_target, w.end) for w in windows])
    assert len(targets) == len(set(targets.tolist()))
    assert targets.min() >= 1 and targets.max() == n - 1
    assert all(w.end - w.start <= ctx and w.first_target > w.start for w in windows)
    if stride < ctx:
        assert set(targets.tolist()) == set(range(1, n))


@pytest.mark.parametrize("bos_id", [None, 999])
def test_scoring_positions_predict_their_targets(bos_id):
    ids = np.arange(100, 160)
    offset = bos_id is not None
    windows = make_windows(len(ids), 8 - offset, 4)
    seen = []
    for batch in iterate_batches(ids, windows, 3, bos_id):
        assert batch.inputs.shape[1] <= 8
        assert (batch.inputs[batch.rows, batch.positions + 1] == ids[batch.targets]).all()
        assert (batch.inputs[batch.rows, batch.positions] == ids[batch.targets - 1]).all()
        assert (batch.context >= 1).all()
        if bos_id is not None:
            assert (batch.inputs[:, 0] == bos_id).all()
        seen.extend(batch.targets.tolist())
    assert sorted(seen) == list(range(1, len(ids)))


def test_wikitext_articles_become_documents():
    lines = ["", " = A = \n", "", " text a \n", " = = Section = = \n", " more \n", " = B = \n", " text b \n"]
    documents = split_documents(lines)
    assert len(documents) == 2
    assert documents[0].startswith(" = A = ") and "Section" in documents[0]
    assert documents[1] == " = B = \n text b \n"
