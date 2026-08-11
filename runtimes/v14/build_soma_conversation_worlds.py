#!/usr/bin/env python3
"""build a substantial licensed v13 conversational curriculum."""

import argparse
import gzip
import hashlib
import json
import random
import re
from pathlib import Path

from build_soma_chat_curriculum import (
    CONTROL_PIECES, OASST_LICENSE_URL, OASST_URL, build_fundamentals,
    clean_message, download, node_score, render_dialogue, safe_node,
    validate_protocol, verify_tokenizer, write_dialogues)


SODA_PARQUET_URLS = tuple(
    "https://huggingface.co/api/datasets/allenai/soda/parquet/"
    f"default/train/{index}.parquet"
    for index in range(3))
SODA_CARD_URL = (
    "https://huggingface.co/datasets/allenai/soda/resolve/main/"
    "README.md?download=true")
ASSISTANT_PROFANITY = re.compile(
    r"\b(?:fuck|fucking|shit|bullshit|bitch|bastard|asshole|"
    r"motherfucker|cunt|dickhead|what the hell)\b", re.IGNORECASE)


def _message_paths(tree, max_messages=10, max_paths=6):
    root = tree.get("prompt", {})
    if not safe_node(root, "prompter"):
        return []
    candidates = []

    def visit(node, expected, messages, score):
        if len(messages) >= max_messages or not safe_node(node, expected):
            return
        text = clean_message(node.get("text", ""), 800)
        current = messages + [(expected, text)]
        current_score = score + node_score(node, expected)
        if expected == "assistant":
            candidates.append((len(current), current_score, current))
        next_role = "assistant" if expected == "prompter" else "prompter"
        children = [child for child in node.get("replies", [])
                    if safe_node(child, next_role)]
        children.sort(
            key=lambda child: node_score(child, next_role), reverse=True)
        for child in children[:3]:
            visit(child, next_role, current, current_score)

    visit(root, "prompter", [], 0.0)
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    dialogues = []
    seen = set()
    for _, _, messages in candidates:
        dialogue = tuple(
            (messages[index][1], messages[index + 1][1])
            for index in range(0, len(messages) - 1, 2)
            if (messages[index][0] == "prompter"
                and messages[index + 1][0] == "assistant"))
        key = tuple(text.lower() for pair in dialogue for text in pair)
        if dialogue and key not in seen:
            seen.add(key)
            dialogues.append(dialogue)
        if len(dialogues) >= max_paths:
            break
    return dialogues


def oasst_dialogues(path, limit, rng):
    dialogues = []
    seen = set()
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            for dialogue in _message_paths(json.loads(line)):
                key = tuple(text.lower() for pair in dialogue for text in pair)
                if key in seen:
                    continue
                seen.add(key)
                dialogues.append(dialogue)
    rng.shuffle(dialogues)
    return dialogues[:limit] if limit else dialogues


def _soda_dialogue(values):
    if not isinstance(values, list) or len(values) < 2:
        return None
    values = values[:12]
    if len(values) % 2:
        values = values[:-1]
    dialogue = []
    for index in range(0, len(values), 2):
        user = clean_message(values[index], 500)
        soma = clean_message(values[index + 1], 800)
        if not user or not soma:
            return None
        if ASSISTANT_PROFANITY.search(soma):
            return None
        combined = f"{user}\n{soma}".lower()
        if ("```" in combined or "http://" in combined
                or "https://" in combined):
            return None
        if not (2 <= len(user.split()) <= 90
                and 2 <= len(soma.split()) <= 120):
            return None
        dialogue.append((user, soma))
    return tuple(dialogue) if dialogue else None


def soda_dialogues(paths, limit, rng):
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "building soda requires pyarrow: python -m pip install pyarrow"
        ) from error
    reservoir = []
    valid_seen = 0
    seen_keys = set()
    columns = ["dialogue", "pmi_relation_tail_answer"]
    for path in paths:
        source = parquet.ParquetFile(path)
        for batch in source.iter_batches(batch_size=4096, columns=columns):
            dialogues = batch.column(0).to_pylist()
            quality = batch.column(1).to_pylist()
            for values, answer in zip(dialogues, quality):
                if answer != "Yes":
                    continue
                dialogue = _soda_dialogue(values)
                if not dialogue:
                    continue
                key = hashlib.sha1(
                    "\n".join(
                        text.lower() for pair in dialogue for text in pair)
                    .encode("utf-8")).digest()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                valid_seen += 1
                if not limit or len(reservoir) < limit:
                    reservoir.append(dialogue)
                else:
                    replacement = rng.randrange(valid_seen)
                    if replacement < limit:
                        reservoir[replacement] = dialogue
    rng.shuffle(reservoir)
    return reservoir, valid_seen


def stats_for_dialogues(dialogues):
    return {
        "dialogues": len(dialogues),
        "turns": sum(len(dialogue) for dialogue in dialogues),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path.home() / "Library/Application Support/soma/data")
    parser.add_argument("--fundamentals", type=int, default=12000)
    parser.add_argument("--fundamentals-mix", type=int, default=3000)
    parser.add_argument("--oasst", type=int, default=12000)
    parser.add_argument(
        "--soda", type=int, default=0,
        help="maximum soda dialogues; zero keeps every valid dialogue")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / ".conversation_source_cache"
    oasst_archive = cache / "oasst1_ready_trees.jsonl.gz"
    oasst_license = cache / "oasst1_LICENSE"
    soda_parquets = tuple(
        cache / f"soda_train_{index}.parquet" for index in range(3))
    soda_card = cache / "soda_README.md"
    if not args.offline:
        download(OASST_URL, oasst_archive)
        download(OASST_LICENSE_URL, oasst_license)
        for url, path in zip(SODA_PARQUET_URLS, soda_parquets):
            download(url, path)
        download(SODA_CARD_URL, soda_card)
    for required in (
            oasst_archive, oasst_license, *soda_parquets, soda_card):
        if not required.exists():
            raise FileNotFoundError(required)

    fundamentals_rng = random.Random(args.seed)
    fundamentals = list(build_fundamentals(
        args.fundamentals, fundamentals_rng))
    fundamentals_mix = fundamentals[:max(
        0, min(args.fundamentals_mix, len(fundamentals)))]

    oasst_rng = random.Random(args.seed + 1)
    oasst = oasst_dialogues(oasst_archive, args.oasst, oasst_rng)
    oasst_rendered = [
        render_dialogue(dialogue, oasst_rng, index)
        for index, dialogue in enumerate(oasst)]

    soda_rng = random.Random(args.seed + 2)
    soda, soda_valid = soda_dialogues(soda_parquets, args.soda, soda_rng)
    soda_rendered = [
        render_dialogue(dialogue, soda_rng, args.oasst + index)
        for index, dialogue in enumerate(soda)]

    foundations_path = args.out_dir / "conversation_foundations_v4.txt"
    assistant_path = args.out_dir / "conversation_human_assistant_v4.txt"
    worlds_path = args.out_dir / "conversation_worlds_v4.txt"
    foundation_stats = write_dialogues(foundations_path, fundamentals)
    assistant_stats = write_dialogues(
        assistant_path, fundamentals_mix + oasst_rendered)
    mixed = fundamentals_mix + oasst_rendered + soda_rendered
    random.Random(args.seed + 3).shuffle(mixed)
    worlds_stats = write_dialogues(worlds_path, mixed)
    for path in (foundations_path, assistant_path, worlds_path):
        validate_protocol(path)

    tokenizer_model = (
        Path(__file__).resolve().parent / "tokenizer" / "soma-1k-v1.model")
    manifest = {
        "name": "soma conversation worlds v4",
        "protocol": "v13 compact role and turn boundaries",
        "seed": args.seed,
        "files": {
            foundations_path.name: foundation_stats,
            assistant_path.name: assistant_stats,
            worlds_path.name: worlds_stats,
        },
        "components": {
            "fundamentals": {
                "dialogues": foundation_stats["dialogues"],
                "turns": foundation_stats["turns"],
                "dialogues_in_mixed_files": len(fundamentals_mix),
            },
            "oasst1": stats_for_dialogues(oasst),
            "soda": stats_for_dialogues(soda),
            "soda_valid_rows_seen": soda_valid,
        },
        "sources": [
            {
                "name": "OpenAssistant OASST1",
                "url": OASST_URL,
                "license": "Apache-2.0",
                "license_file": str(oasst_license),
            },
            {
                "name": "AllenAI SODA",
                "urls": list(SODA_PARQUET_URLS),
                "license": "CC-BY-4.0",
                "attribution_file": str(soda_card),
            },
        ],
        "tokenizer_check": verify_tokenizer(tokenizer_model),
        "notes": [
            "recorded language is separated only by native role and turn pieces",
            "no synthetic state prelude occupies the conversational context",
            "oasst paths are human-written and quality filtered",
            "soda supplies synthetic but broad social turn structure",
            "unsafe, deleted, non-english, code-heavy, url-bearing and very long records are excluded",
        ],
    }
    manifest_path = args.out_dir / "conversation_worlds_v4_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    readme = args.out_dir / "conversation_worlds_v4_readme.txt"
    readme.write_text(
        "soma conversation worlds v4\n\n"
        "train conversation_foundations_v4.txt first when a checkpoint does "
        "not yet follow turns reliably. continue with "
        "conversation_human_assistant_v4.txt for direct assistant behaviour. "
        "use conversation_worlds_v4.txt for the complete mixed pass.\n\n"
        "use soma v13. the corpus uses compact role boundaries. normal native "
        "feedback is compatible with turn learning; use dense updates for the "
        "first pass and save under a new checkpoint name.\n\n"
        "sources and licenses are recorded in conversation_worlds_v4_manifest.json.\n",
        encoding="utf-8")

    for path, stats in (
            (foundations_path, foundation_stats),
            (assistant_path, assistant_stats),
            (worlds_path, worlds_stats)):
        print(
            f"wrote {path} · {stats['dialogues']:,} dialogues · "
            f"{stats['turns']:,} turns · {stats['bytes']/1024/1024:.1f} mb")
    print(f"wrote {manifest_path}")
    print(f"wrote {readme}")


if __name__ == "__main__":
    main()
