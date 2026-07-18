from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

from minichat.tokenizer import build_tokenizer, normalize_text


def iter_documents(paths: list[str], max_documents: int) -> Iterator[str]:
    yielded = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str):
                    continue
                text = normalize_text(text)
                if len(text) < 100:
                    continue
                yield text
                yielded += 1
                if max_documents and yielded >= max_documents:
                    return


def expand_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        paths.extend(matches or ([pattern] if Path(pattern).is_file() else []))
    paths = sorted(set(str(Path(path).resolve()) for path in paths))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {patterns}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level Chinese BPE tokenizer.")
    parser.add_argument("--input", nargs="+", required=True, help="Parquet paths or glob patterns")
    parser.add_argument("--output", default="tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--max-documents", type=int, default=100_000)
    args = parser.parse_args()

    paths = expand_inputs(args.input)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer, trainer = build_tokenizer(args.vocab_size)
    tokenizer.train_from_iterator(
        iter_documents(paths, args.max_documents),
        trainer=trainer,
        length=args.max_documents or None,
    )
    tokenizer.save(str(output / "tokenizer.json"))
    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "sources": paths,
        "max_documents": args.max_documents,
        "special_tokens": {
            token: tokenizer.token_to_id(token)
            for token in ("<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>")
        },
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Saved {tokenizer.get_vocab_size():,}-token vocabulary to {output}")


if __name__ == "__main__":
    main()

