#!/usr/bin/env python3
"""turn ordinary question-answer data into v12.3-native training text."""

import argparse
import csv
import json
import random
import subprocess
import zipfile
from pathlib import Path

from build_soma_chat_curriculum import clean_message, render_dialogue

OPENBOOK_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"


def clean(text):
    return clean_message(text, 1200)


def record(question, answer, rng=None, index=0):
    question, answer = clean(question), clean(answer)
    if not question or not answer:
        return None
    if not question.endswith(("?", ".", "!", ":")):
        question += "?"
    if not answer.endswith((".", "!", "?")):
        answer += "."
    return render_dialogue(
        ((question, answer),), rng or random.Random(12345), index)


def read_csv(path, question_column, answer_column, rng):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if question_column not in fields or answer_column not in fields:
            raise ValueError(
                f"expected columns {question_column!r} and {answer_column!r}; "
                f"found {fields}")
        for index, row in enumerate(reader):
            item = record(
                row[question_column], row[answer_column], rng, index)
            if item:
                yield item


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        print(f"downloading {destination.name}")
        partial = destination.with_suffix(destination.suffix + ".partial")
        try:
            subprocess.run(
                ["/usr/bin/curl", "--fail", "--location", "--retry", "3",
                 "--output", str(partial), url],
                check=True)
            partial.replace(destination)
        finally:
            partial.unlink(missing_ok=True)
    return destination


def multiple_choice_records(archive_path, prefixes, rng, index_offset=0):
    index = index_offset
    with zipfile.ZipFile(archive_path) as archive:
        names = [name for name in archive.namelist()
                 if name.endswith(".jsonl") and any(
                     prefix in name for prefix in prefixes)]
        for name in sorted(names):
            for line in archive.read(name).decode("utf-8").splitlines():
                data = json.loads(line)
                question = data.get("question", {})
                choices = question.get("choices", [])
                answer_key = data.get("answerKey")
                answer = next((choice.get("text", "") for choice in choices
                               if choice.get("label") == answer_key), "")
                item = record(
                    question.get("stem", ""), answer, rng, index)
                if item:
                    yield item
                    index += 1


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item in records:
            f.write(item)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="csv with question and answer columns")
    parser.add_argument("--out-dir", type=Path,
                        default=Path.home() / "Library/Application Support/soma/data")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--download-common-knowledge", action="store_true")
    args = parser.parse_args()

    csv_rng = random.Random(12345)
    csv_output = args.out_dir / f"{args.csv.stem}_soma_qa.txt"
    csv_count = write_records(
        csv_output,
        read_csv(
            args.csv, args.question_column, args.answer_column, csv_rng))
    print(f"wrote {csv_output} ({csv_count:,} q&a turns)")

    if not args.download_common_knowledge:
        return

    cache = args.out_dir / ".qa_source_cache"
    openbook = download(OPENBOOK_URL, cache / "OpenBookQA-V1-Sep2018.zip")
    common_output = args.out_dir / "soma_common_knowledge_qa.txt"

    def all_records():
        rng = random.Random(12346)
        yield from read_csv(
            args.csv, args.question_column, args.answer_column, rng)
        yield from multiple_choice_records(
            openbook, ("Data/Main/",), rng, index_offset=csv_count)

    common_count = write_records(common_output, all_records())
    manifest = args.out_dir / "soma_common_knowledge_qa_manifest.txt"
    manifest.write_text(
        "soma common knowledge q&a\n"
        f"records: {common_count}\n"
        "format: v12.3.11 compact user -> turn -> soma -> answer -> turn\n"
        "sources: user csv; ai2 openbookqa\n"
        "downloads retained in .qa_source_cache for provenance.\n",
        encoding="utf-8")
    print(f"wrote {common_output} ({common_count:,} q&a turns)")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
