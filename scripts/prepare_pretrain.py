from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from minichat.tokenizer import load_tokenizer, normalize_text


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path).resolve() for path in glob.glob(pattern, recursive=True))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No parquet files matched: {patterns}")
    return paths


def save_state(path: Path, values: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(values, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize FineWeb parquet files into uint16 tokens.")
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=0, help="0 means no limit")
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    paths = expand_inputs(args.input)
    tokenizer = load_tokenizer(args.tokenizer)
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError("Vocabulary does not fit in uint16")
    eos_id = tokenizer.token_to_id("<|eos|>")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part")
    state_path = output.with_suffix(output.suffix + ".state.json")
    state = {"file_index": 0, "row_group": 0, "tokens": 0, "documents": 0}
    if state_path.exists() and part.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state.update(json.load(handle))
        print(f"Resuming at file {state['file_index']}, row group {state['row_group']}")
    else:
        part.unlink(missing_ok=True)

    stop = False
    with part.open("ab") as output_handle:
        for file_index in range(int(state["file_index"]), len(paths)):
            parquet = pq.ParquetFile(paths[file_index])
            first_group = int(state["row_group"]) if file_index == int(state["file_index"]) else 0
            progress = tqdm(
                range(first_group, parquet.metadata.num_row_groups),
                desc=paths[file_index].name,
                unit="group",
            )
            for row_group in progress:
                table = parquet.read_row_group(row_group, columns=["text"])
                texts = [
                    normalized
                    for text in table.column(0).to_pylist()
                    if isinstance(text, str) and len(normalized := normalize_text(text)) >= args.min_chars
                ]
                for offset in range(0, len(texts), args.batch_size):
                    encodings = tokenizer.encode_batch(texts[offset : offset + args.batch_size])
                    chunks = [encoding.ids + [eos_id] for encoding in encodings]
                    if not chunks:
                        continue
                    tokens = np.fromiter(
                        (token for chunk in chunks for token in chunk), dtype=np.uint16
                    )
                    if args.max_tokens:
                        remaining = args.max_tokens - int(state["tokens"])
                        if remaining <= 0:
                            stop = True
                            break
                        tokens = tokens[:remaining]
                    tokens.tofile(output_handle)
                    state["tokens"] = int(state["tokens"]) + len(tokens)
                    state["documents"] = int(state["documents"]) + len(chunks)
                    if args.max_tokens and int(state["tokens"]) >= args.max_tokens:
                        stop = True
                        break
                output_handle.flush()
                state["file_index"] = file_index
                state["row_group"] = row_group + 1
                save_state(state_path, state)
                progress.set_postfix(tokens=f"{int(state['tokens']):,}")
                if stop:
                    break
            if stop:
                break
            state["file_index"] = file_index + 1
            state["row_group"] = 0
            save_state(state_path, state)

    os.replace(part, output)
    state_path.unlink(missing_ok=True)
    metadata = {
        "tokens": int(state["tokens"]),
        "documents": int(state["documents"]),
        "dtype": "uint16",
        "vocab_size": tokenizer.get_vocab_size(),
        "sources": [str(path) for path in paths],
        "source_dataset": "HuggingFaceFW/fineweb-2",
        "source_config": "cmn_Hani",
        "source_license": "ODC-BY-1.0",
    }
    with output.with_suffix(".meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {metadata['tokens']:,} tokens from {metadata['documents']:,} documents to {output}")


if __name__ == "__main__":
    main()

