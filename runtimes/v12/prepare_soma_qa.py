#!/usr/bin/env python3
"""turn ordinary question-answer data into soma turn-protocol training text.

each answer is preceded by a compact soma-state block. this teaches the
placement and role of the live state prelude without fabricating a fixed
learning rate, byte count, or other changing telemetry for every example.
"""

import argparse
import csv
import html
import json
import re
import subprocess
import unicodedata
import zipfile
from pathlib import Path


TURN = "\x1e"
STATE = """<soma_state>
source: chat
architecture: serial trace mlp
mode: answer
</soma_state>"""

OPENBOOK_URL = "https://s3-us-west-2.amazonaws.com/ai2-website/data/OpenBookQA-V1-Sep2018.zip"


def clean(text):
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace(TURN, " ")
    return re.sub(r"\s+", " ", text).strip()


def record(question, answer):
    question, answer = clean(question), clean(answer)
    if not question or not answer:
        return None
    if not question.endswith(("?", ".", "!", ":")):
        question += "?"
    if not answer.endswith((".", "!", "?")):
        answer += "."
    return f"you: {question}{TURN}\n{STATE}{TURN}\nsoma: {answer}{TURN}\n"


def read_csv(path, question_column, answer_column):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if question_column not in fields or answer_column not in fields:
            raise ValueError(
                f"expected columns {question_column!r} and {answer_column!r}; "
                f"found {fields}")
        for row in reader:
            item = record(row[question_column], row[answer_column])
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


def multiple_choice_records(archive_path, prefixes):
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
                item = record(question.get("stem", ""), answer)
                if item:
                    yield item


def write_records(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="ascii", newline="\n") as f:
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

    csv_output = args.out_dir / f"{args.csv.stem}_soma_qa.txt"
    csv_count = write_records(
        csv_output,
        read_csv(args.csv, args.question_column, args.answer_column))
    print(f"wrote {csv_output} ({csv_count:,} q&a turns)")

    if not args.download_common_knowledge:
        return

    cache = args.out_dir / ".qa_source_cache"
    openbook = download(OPENBOOK_URL, cache / "OpenBookQA-V1-Sep2018.zip")
    common_output = args.out_dir / "soma_common_knowledge_qa.txt"

    def all_records():
        yield from read_csv(args.csv, args.question_column, args.answer_column)
        yield from multiple_choice_records(openbook, ("Data/Main/",))

    common_count = write_records(common_output, all_records())
    manifest = args.out_dir / "soma_common_knowledge_qa_manifest.txt"
    manifest.write_text(
        "soma common knowledge q&a\n"
        f"records: {common_count}\n"
        "format: you -> compact soma_state -> soma -> record separator\n"
        "sources: user csv; ai2 openbookqa\n"
        "downloads retained in .qa_source_cache for provenance.\n",
        encoding="utf-8")
    print(f"wrote {common_output} ({common_count:,} q&a turns)")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
