from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from minichat.tokenizer import load_tokenizer, normalize_text


SYSTEM_PROMPT = "你是一个友善、诚实、有帮助的中文助手。回答应当清楚、准确; 不知道时直接说明。"


def json_lines(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_coig(
    directory: Path, include_code: bool, chat_only: bool = False
) -> Iterator[tuple[str, list[tuple[str, str]]]]:
    tar_path = directory / "counterfactural_correction_multi_round_chat.tar.gz"
    with tarfile.open(tar_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            data = json.loads(extracted.read().decode("utf-8"))
            turns: list[tuple[str, str]] = []
            for round_index in range(10):
                item = data.get(f"round_{round_index}")
                if not item:
                    continue
                try:
                    response = json.loads(item["response"])
                    turns.append((response["Q"], response["A"]))
                except (KeyError, json.JSONDecodeError, TypeError):
                    continue
            if turns:
                yield "", turns

    if chat_only:
        return

    for filename in ("exam_instructions.jsonl", "human_value_alignment_instructions_part2.json"):
        for data in json_lines(directory / filename):
            instruction = data.get("textbox_q_instruction", "")
            question = "\n".join(
                value
                for value in (data.get("textbox_q_context", ""), data.get("textbox_question", ""))
                if value
            )
            answer = data.get("textbox_answer_analysis") or data.get("textbox_answer", "")
            if question and answer:
                yield instruction, [(question, answer)]

    with (directory / "human_value_alignment_instructions_part1.json").open(
        "r", encoding="utf-8"
    ) as handle:
        for data in json.load(handle):
            question = data.get("input") or data.get("instruction", "")
            instruction = data.get("instruction", "") if data.get("input") else ""
            answer = data.get("output", "")
            if question and answer:
                yield instruction, [(question, answer)]

    for data in json_lines(directory / "translated_instructions.jsonl"):
        question = data.get("trans_input") or data.get("trans_instruction", "")
        instruction = data.get("trans_instruction", "") if data.get("trans_input") else ""
        answer = data.get("trans_output", "")
        if question and answer:
            yield instruction, [(question, answer)]

    if include_code:
        for data in json_lines(directory / "leetcode_instructions.jsonl"):
            question = data.get("input") or data.get("instruction", "")
            instruction = data.get("instruction", "") if data.get("input") else ""
            answer = data.get("output", "")
            if question and answer:
                yield instruction, [(question, answer)]


def append_masked(tokenizer: Tokenizer, ids: list[int], labels: list[int], text: str) -> None:
    values = tokenizer.encode(text).ids
    ids.extend(values)
    labels.extend([-100] * len(values))


def encode_chat(
    tokenizer: Tokenizer,
    instruction: str,
    turns: list[tuple[str, str]],
    width: int,
) -> tuple[list[int], list[int]] | None:
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    system_id = tokenizer.token_to_id("<|system|>")
    user_id = tokenizer.token_to_id("<|user|>")
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    ids = [bos_id, system_id]
    labels = [-100, -100]
    system = SYSTEM_PROMPT
    if instruction:
        system += "\n任务说明: " + normalize_text(instruction)
    append_masked(tokenizer, ids, labels, "\n" + system + "\n")
    for question, answer in turns:
        ids.append(user_id)
        labels.append(-100)
        append_masked(tokenizer, ids, labels, "\n" + normalize_text(question) + "\n")
        ids.append(assistant_id)
        labels.append(-100)
        answer_ids = tokenizer.encode("\n" + normalize_text(answer)).ids + [eos_id]
        ids.extend(answer_ids)
        labels.extend(answer_ids)
    if sum(label != -100 for label in labels) < 2:
        return None
    if len(ids) > width:
        # Preserve the answer end; left truncation is preferable to discarding the target.
        ids = ids[-width:]
        labels = labels[-width:]
        ids[0] = bos_id
        labels[0] = -100
    return ids, labels


@dataclass
class RecordWriter:
    directory: Path
    width: int

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ids_handle = (self.directory / "input_ids.bin.part").open("wb")
        self.labels_handle = (self.directory / "labels.bin.part").open("wb")
        self.records = 0
        self.assistant_tokens = 0

    def write(self, ids: list[int], labels: list[int], pad_id: int) -> None:
        padding = self.width - len(ids)
        ids = ids + [pad_id] * padding
        labels = labels + [-100] * padding
        np.asarray(ids, dtype=np.uint16).tofile(self.ids_handle)
        np.asarray(labels, dtype=np.int32).tofile(self.labels_handle)
        self.records += 1
        self.assistant_tokens += sum(label != -100 for label in labels)

    def close(self, metadata: dict) -> None:
        self.ids_handle.close()
        self.labels_handle.close()
        os.replace(self.directory / "input_ids.bin.part", self.directory / "input_ids.bin")
        os.replace(self.directory / "labels.bin.part", self.directory / "labels.bin")
        metadata.update(
            records=self.records,
            assistant_tokens=self.assistant_tokens,
            seq_len=self.width - 1,
        )
        with (self.directory / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COIG into masked chat records.")
    parser.add_argument("--input", default="data/raw/coig")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json")
    parser.add_argument("--output", default="data/processed/sft")
    parser.add_argument("--validation-output", default="data/processed/sft-validation")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--validation-percent", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--include-code", action="store_true")
    parser.add_argument(
        "--chat-only", action="store_true", help="Use only COIG multi-round chat records"
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    width = args.seq_len + 1
    train_writer = RecordWriter(Path(args.output), width)
    val_writer = RecordWriter(Path(args.validation_output), width)
    pad_id = tokenizer.token_to_id("<|pad|>")
    seen: set[bytes] = set()
    accepted = 0
    try:
        for instruction, turns in tqdm(
            read_coig(Path(args.input), args.include_code, args.chat_only), desc="COIG"
        ):
            fingerprint = hashlib.blake2b(
                json.dumps([instruction, turns], ensure_ascii=False).encode("utf-8"), digest_size=16
            ).digest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            encoded = encode_chat(tokenizer, instruction, turns, width)
            if encoded is None:
                continue
            bucket = int.from_bytes(fingerprint[:2], "little") % 100
            writer = val_writer if bucket < args.validation_percent else train_writer
            writer.write(*encoded, pad_id)
            accepted += 1
            if args.max_examples and accepted >= args.max_examples:
                break
    finally:
        common = {
            "source_dataset": "BAAI/COIG",
            "source_license": "Apache-2.0 (see upstream dataset card for component licenses)",
            "tokenizer_vocab_size": tokenizer.get_vocab_size(),
            "system_prompt": SYSTEM_PROMPT,
        }
        train_writer.close({**common, "split": "train"})
        val_writer.close({**common, "split": "validation"})
    print(f"Wrote {train_writer.records:,} train and {val_writer.records:,} validation records")


if __name__ == "__main__":
    main()
