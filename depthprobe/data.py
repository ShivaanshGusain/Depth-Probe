"""Local corpora, document boundaries, and sliding-window scoring.

Every target token is scored at most once. Logits at input position p predict token p + 1.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# WikiText article titles look like " = Title = "; section headings use " = = ".
WIKITEXT_ARTICLE = re.compile(r"^ = [^=].* = $")
# Non-WikiText files: blocks of lines become pseudo-documents for the bootstrap.
FALLBACK_LINES = 200


@dataclass
class Corpus:
    ids: np.ndarray  # int64 token ids, documents concatenated
    doc: np.ndarray  # int32 document index of each token
    n_docs: int


@dataclass(frozen=True)
class Window:
    start: int
    end: int
    first_target: int


@dataclass
class Batch:
    inputs: np.ndarray  # [B, T] int64
    rows: np.ndarray  # batch row of each scored token
    positions: np.ndarray  # input position whose logits predict the target
    targets: np.ndarray  # index of the target in Corpus.ids
    context: np.ndarray  # corpus tokens visible before the target (BOS excluded)


def read_lines(path, split):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.rglob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No {split}-*.parquet under {path}")
    else:
        files = [path]
    lines = []
    for file in files:
        if file.suffix == ".parquet":
            import pyarrow.parquet as pq

            lines += pq.read_table(file, columns=["text"]).column("text").to_pylist()
        elif file.name.endswith((".jsonl", ".json", ".jsonl.gz", ".json.gz")):
            # JSON lines with a "text" field, e.g. C4 shards (c4-validation.00000-of-00008.json.gz). Each record
            # becomes one entry; without WikiText headings, blocks of records form the bootstrap documents.
            import gzip
            import json

            opener = gzip.open if file.suffix == ".gz" else open
            with opener(file, "rt", encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        lines.append(json.loads(line)["text"].rstrip("\n") + "\n")
                    except (json.JSONDecodeError, KeyError, TypeError) as error:
                        raise ValueError(f"{file}:{number}: expected JSON lines, one object with a 'text' field "
                                         "per line (a single JSON array is not supported)") from error
        else:
            lines += file.read_text(encoding="utf-8").splitlines(keepends=True)
    return lines


def split_documents(lines, joiner=""):
    starts = [i for i, line in enumerate(lines) if WIKITEXT_ARTICLE.match(line.rstrip("\n"))]
    if not starts:
        starts = list(range(0, len(lines), FALLBACK_LINES))
    elif starts[0] != 0:
        starts.insert(0, 0)
    bounds = starts + [len(lines)]
    documents = [joiner.join(lines[a:b]) for a, b in zip(bounds, bounds[1:])]
    return [document for document in documents if document.strip()]


def load_corpus(tokenizer, path, split, *, max_tokens=None, joiner="", shuffle_seed=None):
    documents = split_documents(read_lines(path, split), joiner)
    if shuffle_seed is not None:
        order = np.random.default_rng(shuffle_seed).permutation(len(documents))
        documents = [documents[i] for i in order]
    ids, doc = [], []
    for index, text in enumerate(documents):
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids.extend(tokens)
        doc.extend([index] * len(tokens))
        if max_tokens is not None and len(ids) > max_tokens:
            break
    if max_tokens is not None:
        # max_tokens + 1 inputs give max_tokens possible targets.
        ids, doc = ids[: max_tokens + 1], doc[: max_tokens + 1]
    doc = np.asarray(doc, dtype=np.int32)
    return Corpus(np.asarray(ids, dtype=np.int64), doc, int(np.unique(doc).size))


def make_windows(n_tokens, ctx, stride):
    if not 0 < stride <= ctx:
        raise ValueError("Need 0 < stride <= ctx")
    windows, start, scored_until = [], 0, 1
    while True:
        end = min(start + ctx, n_tokens)
        first = max(scored_until, start + 1)
        if first < end:
            windows.append(Window(start, end, first))
        scored_until = end
        if end >= n_tokens:
            return windows
        start += stride


def iterate_batches(ids, windows, batch_size, bos_id=None):
    offset = 0 if bos_id is None else 1
    by_length = {}
    for window in windows:
        by_length.setdefault(window.end - window.start, []).append(window)
    for length, group in by_length.items():
        for i in range(0, len(group), batch_size):
            chunk = group[i : i + batch_size]
            inputs = np.empty((len(chunk), length + offset), dtype=np.int64)
            rows, positions, targets, context = [], [], [], []
            for row, window in enumerate(chunk):
                if offset:
                    inputs[row, 0] = bos_id
                inputs[row, offset:] = ids[window.start : window.end]
                target = np.arange(window.first_target, window.end)
                rows.append(np.full(target.size, row))
                positions.append(target - window.start - 1 + offset)
                targets.append(target)
                context.append(target - window.start)
            yield Batch(inputs, *(np.concatenate(x) for x in (rows, positions, targets, context)))
