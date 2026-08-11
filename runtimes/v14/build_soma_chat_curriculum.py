#!/usr/bin/env python3
"""build a compact, soma-native conversational curriculum.

the output mirrors the v13 runtime exactly:

  <|user|>prompt<|turn|><|soma|>reply<|turn|>

the fundamentals file teaches ordinary turn behaviour through deterministic
micro-dialogues. the full curriculum adds short, high-quality english paths
from the human-written openassistant oasst1 conversation trees.
"""

import argparse
import gzip
import hashlib
import html
import json
import random
import re
import subprocess
import unicodedata
from pathlib import Path


OASST_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
    "2023-04-12_oasst_ready.trees.jsonl.gz?download=true")
OASST_LICENSE_URL = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
    "LICENSE?download=true")
CONTROL_PIECES = (
    "<|soma_state|>", "<|turn|>", "<|user|>", "<|soma|>", "<|dream|>")

NAMES = (
    "james", "alex", "mia", "sam", "maya", "leo", "kim", "riley",
    "noah", "ella", "ben", "zoe", "lucy", "daniel", "sophie", "max")
OBJECTS = (
    "blue notebook", "silver key", "green mug", "old photograph",
    "small radio", "red scarf", "wooden box", "train ticket",
    "yellow umbrella", "glass marble", "paper map", "black guitar")
PLACES = (
    "kitchen drawer", "hallway table", "top shelf", "garden shed",
    "desk", "backpack", "bedside cabinet", "front room")
ACTIVITIES = (
    "reading", "walking", "cooking", "making music", "drawing",
    "learning something new", "watching the rain", "talking with a friend")
MOODS = (
    ("good", "i'm glad to hear that."),
    ("tired", "that sounds tiring. i hope you get some rest."),
    ("nervous", "that makes sense. we can take it one step at a time."),
    ("excited", "that's lovely. what are you excited about?"),
    ("a little lost", "that's all right. tell me what feels unclear."),
    ("calm", "that sounds peaceful."))

DIRECT_PAIRS = (
    ("hello.", "hello. how are you?"),
    ("hi there.", "hi. it's good to hear from you."),
    ("how are you?", "i'm here and ready to talk. how are you?"),
    ("good morning.", "good morning. how is your day starting?"),
    ("good night.", "good night. i hope you sleep well."),
    ("thank you.", "you're welcome."),
    ("are you listening?", "yes, i'm listening."),
    ("can we talk?", "of course. what would you like to talk about?"),
    ("i don't understand.", "that's okay. tell me which part is unclear."),
    ("what do you mean?", "i can say it more simply. which part should i explain?"),
    ("please be brief.", "of course. i'll keep it short."),
    ("tell me honestly.", "i will answer as clearly as i can."),
    ("i made a mistake.", "that's all right. mistakes can be corrected."),
    ("i changed my mind.", "that's okay. what do you think now?"),
    ("are you still there?", "yes, i'm still here."),
    ("what should we do next?", "tell me the goal, and we can choose the next step."),
    ("can you repeat that?", "yes. i'll say it again more clearly."),
    ("that wasn't my question.", "you're right. ask it again, and i'll answer directly."),
    ("i need a moment to think.", "take your time."),
    ("bye for now.", "goodbye. talk to you later."))

KNOWLEDGE = (
    ("why is the sky blue?",
     "sunlight scatters in the atmosphere, and blue light scatters more strongly than most other visible colours."),
    ("what do plants need to grow?",
     "most plants need light, water, air, nutrients, and a suitable temperature."),
    ("why do we sleep?",
     "sleep helps the brain and body recover, regulate themselves, and consolidate memory."),
    ("what is gravity?",
     "gravity is the attraction between objects with mass."),
    ("what is a year?",
     "a year is approximately the time earth takes to orbit the sun."),
    ("why does ice melt?",
     "ice melts when it absorbs enough heat for its solid structure to become liquid water."),
    ("what does a compass do?",
     "a compass uses earth's magnetic field to indicate direction."),
    ("what is a library?",
     "a library is a place or collection where books and other information are organized for people to use."),
    ("what makes a question useful?",
     "a useful question makes the uncertainty clear enough to investigate."),
    ("what is a conversation?",
     "a conversation is an exchange in which people listen, respond, and build meaning together."))

REPAIR_DIALOGUES = (
    (("tell me about the bank.", "do you mean a financial bank or the side of a river?"),
     ("the side of a river.", "a river bank is the land along the edge of a river.")),
    (("is it close?", "what place are you asking about?"),
     ("the train station.", "i don't know your location, so i can't tell how close the station is.")),
    (("he took it there.", "who does 'he' refer to, and what did he take?"),
     ("leo took the key to the kitchen.", "understood. leo took the key to the kitchen.")),
    (("give me the answer.", "what question would you like me to answer?"),
     ("what is two plus three?", "two plus three is five.")),
    (("no, that's not what i meant.", "thanks for correcting me. what did you mean?"),
     ("i meant the second idea.", "understood. let's focus on the second idea.")),
    (("reminding us of what?", "i need the sentence before that to know what 'that' refers to."),
     ("the story said friendship takes effort.", "then it was reminding us that friendship takes effort.")))

IDENTITY_DIALOGUES = (
    (("what are you?", "i'm a soma model, learning and running locally on this computer."),),
    (("are you finished learning?", "no. my development can continue through new experience."),),
    (("do you know everything?", "no. my knowledge is limited, and i can be mistaken."),),
    (("what have you read?", "i can only answer from the experience preserved in my current model."),),
    (("can you learn my name?", "yes. tell me your name, and i'll pay attention to it."),),
    (("why did you say that?", "i formed that reply from my current state and what i have learned, but i may not be able to explain every internal step."),),
    (("are you a person?", "no. i'm a learning model, not a human person."),),
    (("where are you running?", "i'm running locally on this computer."),))


def clean_message(value, max_chars):
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFC", text)
    for piece in CONTROL_PIECES:
        text = text.replace(piece, " ")
    text = text.replace("\x1e", " ").replace("\x00", " ")
    text = re.sub(r"[\u0001-\u0008\u000b\u000c\u000e-\u001f\u007f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return ""
    return text


def render_dialogue(dialogue, rng=None, dialogue_index=0):
    """render the same compact role boundary used by live v13 chat."""
    chunks = []
    for user, soma in dialogue:
        user = clean_message(user, 500)
        soma = clean_message(soma, 800)
        if not user or not soma:
            return ""
        chunks.append(
            f"<|user|>{user}<|turn|><|soma|>{soma}<|turn|>\n")
    return "".join(chunks)


def greeting_dialogue(rng):
    name = rng.choice(NAMES)
    mood, response = rng.choice(MOODS)
    hello = rng.choice(("hello", "hi", "hello there", "good morning"))
    return (
        (f"{hello}. my name is {name}. how are you?",
         f"{hello}, {name}. i'm here and ready to talk. how are you?"),
        (f"i'm feeling {mood} today.", response),
        ("do you remember my name?", f"yes. you said your name is {name}."))


def memory_dialogue(rng):
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    place = rng.choice(PLACES)
    activity = rng.choice(ACTIVITIES)
    return (
        (f"my name is {name}. i put the {obj} in the {place} before {activity}.",
         f"understood. you put the {obj} in the {place}."),
        ("where did i put it?", f"you put the {obj} in the {place}."),
        ("and what was i about to do?", f"you were about to spend time {activity}."))


def preference_dialogue(rng):
    name = rng.choice(NAMES)
    first, second = rng.sample(ACTIVITIES, 2)
    return (
        (f"{name} likes {first}, but prefers {second}.",
         f"all right. {name} prefers {second}."),
        (f"which activity does {name} prefer?", f"{name} prefers {second}."),
        ("how do you know?", f"you told me that {name} likes {first} but prefers {second}."))


def direct_dialogue(rng):
    question, answer = rng.choice(DIRECT_PAIRS)
    followups = (
        ("did you understand me?", "yes, i understood you."),
        ("can you keep listening?", "yes. go on."),
        ("is that your answer?", "yes. that is my answer."),
        ("can we continue?", "yes. what would you like to say next?"))
    if rng.random() < 0.55:
        return ((question, answer), rng.choice(followups))
    return ((question, answer),)


def knowledge_dialogue(rng):
    question, answer = rng.choice(KNOWLEDGE)
    simpler = answer.split(",", 1)[0].rstrip(".!?") + "."
    return (
        (question, answer),
        ("can you say that more simply?", simpler),
        ("thank you.", "you're welcome."))


def repair_dialogue(rng):
    return rng.choice(REPAIR_DIALOGUES)


def identity_dialogue(rng):
    dialogue = list(rng.choice(IDENTITY_DIALOGUES))
    if rng.random() < 0.5:
        dialogue.append(("can we keep talking?", "yes. what is on your mind?"))
    return tuple(dialogue)


def build_fundamentals(count, rng):
    builders = (
        greeting_dialogue, memory_dialogue, preference_dialogue,
        direct_dialogue, knowledge_dialogue, repair_dialogue,
        identity_dialogue)
    weights = (0.19, 0.20, 0.13, 0.20, 0.10, 0.12, 0.06)
    for index in range(count):
        builder = rng.choices(builders, weights=weights, k=1)[0]
        yield render_dialogue(builder(rng), rng, index)


def label_value(node, name, default=0.0):
    return float(node.get("labels", {}).get(name, {}).get("value", default))


def safe_node(node, expected_role):
    if node.get("role") != expected_role or node.get("lang") != "en":
        return False
    if node.get("deleted") or node.get("review_result") is False:
        return False
    max_chars = 500 if expected_role == "prompter" else 800
    text = clean_message(node.get("text", ""), max_chars)
    if not text or "```" in text or "http://" in text or "https://" in text:
        return False
    if expected_role == "assistant" and len(text.split()) > 120:
        return False
    if label_value(node, "toxicity") > 0.35:
        return False
    for label in ("hate_speech", "sexual_content", "violence", "spam",
                  "lang_mismatch", "not_appropriate", "pii"):
        if label_value(node, label) > 0.2:
            return False
    lowered = text.lower()
    if expected_role == "assistant":
        identity_claims = (
            "as an ai language model", "as an artificial intelligence",
            "i am an ai", "i'm an ai", "i am a language model",
            "i'm a language model", "i possess some of the key characteristics")
        if any(claim in lowered for claim in identity_claims):
            return False
    return True


def node_score(node, expected_role):
    quality = label_value(node, "quality", 0.45)
    helpfulness = label_value(node, "helpfulness", 0.45)
    replies = sum(
        safe_node(child, "prompter" if expected_role == "assistant" else "assistant")
        for child in node.get("replies", []))
    rank = node.get("rank")
    rank_bonus = 0.35 if rank == 0 else (0.1 if rank is not None else 0.0)
    length = len(clean_message(node.get("text", ""), 1000).split())
    concise = max(0.0, 1.0 - length / 140.0)
    return quality + helpfulness + rank_bonus + 0.12 * replies + 0.1 * concise


def best_path(tree, max_messages=8):
    node = tree.get("prompt", {})
    if not safe_node(node, "prompter"):
        return []
    messages = []
    expected = "prompter"
    while node and len(messages) < max_messages:
        if not safe_node(node, expected):
            break
        messages.append((expected, clean_message(node.get("text", ""), 800)))
        expected = "assistant" if expected == "prompter" else "prompter"
        candidates = [child for child in node.get("replies", [])
                      if safe_node(child, expected)]
        if not candidates:
            break
        node = max(candidates, key=lambda item: node_score(item, expected))
    if len(messages) < 2 or messages[-1][0] != "assistant":
        messages = messages[:-1]
    dialogue = []
    for index in range(0, len(messages) - 1, 2):
        if messages[index][0] == "prompter" and messages[index + 1][0] == "assistant":
            dialogue.append((messages[index][1], messages[index + 1][1]))
    return dialogue


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    partial = destination.with_suffix(destination.suffix + ".partial")
    try:
        subprocess.run(
            ["/usr/bin/curl", "--fail", "--location", "--retry", "3",
             "--output", str(partial), url], check=True)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def oasst_dialogues(path, limit, rng):
    dialogues = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            dialogue = best_path(json.loads(line))
            if dialogue:
                dialogues.append(dialogue)
    rng.shuffle(dialogues)
    return dialogues[:limit] if limit else dialogues


def write_dialogues(path, rendered):
    count = turns = 0
    digest = hashlib.sha256()
    with path.open("wb") as output:
        for item in rendered:
            if not item:
                continue
            data = item.encode("utf-8")
            output.write(data)
            digest.update(data)
            count += 1
            turns += item.count("<|user|>")
    return {"dialogues": count, "turns": turns, "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def validate_protocol(path):
    data = path.read_text(encoding="utf-8")
    counts = {piece: data.count(piece) for piece in CONTROL_PIECES}
    turns = counts["<|user|>"]
    if not turns:
        raise ValueError(f"no dialogue turns in {path}")
    for piece in ("<|user|>", "<|soma|>"):
        if counts[piece] != turns:
            raise ValueError(f"protocol mismatch in {path}: {piece}={counts[piece]}, turns={turns}")
    if counts["<|soma_state|>"] != 0:
        raise ValueError(f"state prelude found in lean protocol: {path}")
    if counts["<|turn|>"] != turns * 2:
        raise ValueError(f"turn marker mismatch in {path}: {counts['<|turn|>']} != {turns * 2}")
    if "<soma_state>" in data or "\x1e" in data:
        raise ValueError(f"legacy protocol found in {path}")
    return counts


def verify_tokenizer(model_path):
    try:
        import sentencepiece as spm
    except ImportError:
        return "sentencepiece unavailable; structural validation only"
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    for piece in CONTROL_PIECES:
        ids = processor.encode(piece, out_type=int)
        if len(ids) != 1 or processor.id_to_piece(ids[0]) != piece:
            raise ValueError(f"{piece} is not one reserved tokenizer piece")
    return "all protocol markers are reserved tokenizer pieces"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path.home() / "Library/Application Support/soma/data")
    parser.add_argument("--core-dialogues", type=int, default=7000)
    parser.add_argument("--oasst-dialogues", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = args.out_dir / ".chat_source_cache"
    archive = cache / "oasst1_ready_trees.jsonl.gz"
    license_path = cache / "oasst1_LICENSE"
    if not args.offline:
        download(OASST_URL, archive)
        download(OASST_LICENSE_URL, license_path)
    if not archive.exists():
        raise FileNotFoundError(
            f"missing {archive}; run without --offline to download it")

    core_rng = random.Random(args.seed)
    fundamentals = list(build_fundamentals(args.core_dialogues, core_rng))
    fundamentals_path = args.out_dir / "soma_chat_fundamentals_v1.txt"
    fundamental_stats = write_dialogues(fundamentals_path, fundamentals)
    validate_protocol(fundamentals_path)

    source_rng = random.Random(args.seed + 1)
    source_dialogues = oasst_dialogues(
        archive, args.oasst_dialogues, source_rng)
    rendered_source = [
        render_dialogue(dialogue, source_rng, args.core_dialogues + index)
        for index, dialogue in enumerate(source_dialogues)]
    combined = fundamentals + rendered_source
    random.Random(args.seed + 2).shuffle(combined)
    curriculum_path = args.out_dir / "soma_chat_curriculum_v1.txt"
    curriculum_stats = write_dialogues(curriculum_path, combined)
    validate_protocol(curriculum_path)

    tokenizer_model = Path(__file__).resolve().parent / "tokenizer" / "soma-1k-v1.model"
    tokenizer_check = verify_tokenizer(tokenizer_model)
    manifest = {
        "name": "soma chat curriculum v1",
        "protocol": "v13 compact user -> turn -> soma -> reply -> turn",
        "seed": args.seed,
        "fundamentals": fundamental_stats,
        "curriculum": curriculum_stats,
        "oasst_dialogues": len(source_dialogues),
        "oasst_source": OASST_URL,
        "oasst_license": "apache-2.0; retained in .chat_source_cache/oasst1_LICENSE",
        "tokenizer_check": tokenizer_check,
        "notes": [
            "responses are deliberately concise",
            "no synthetic state prelude is inserted between role and reply",
            "unsafe, non-english, long, code-heavy, url-bearing and legacy-protocol text is excluded",
            "train fundamentals first, then the full curriculum",
        ],
    }
    manifest_path = args.out_dir / "soma_chat_curriculum_v1_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    readme_path = args.out_dir / "soma_chat_curriculum_v1_readme.txt"
    readme_path.write_text(
        "soma chat curriculum v1\n\n"
        "purpose\n"
        "teach an existing english-capable v13 checkpoint the exact live "
        "chat protocol and ordinary conversational behaviour.\n\n"
        "recommended first pass\n"
        "1. train soma_chat_fundamentals_v1.txt for 2 epochs.\n"
        "2. test several greetings, names, follow-ups, corrections, and "
        "short questions in chat.\n"
        "3. train soma_chat_curriculum_v1.txt for 1 epoch if broader replies "
        "are still needed.\n"
        "4. return to a coherent prose or knowledge corpus after the chat "
        "lesson.\n\n"
        "settings\n"
        "use dense v13 updates unless testing an explicit update-clock "
        "ablation. native dream boundaries are compatible with this protocol; "
        "save to a new checkpoint name.\n\n"
        "notes\n"
        "the full curriculum already contains the fundamentals. it is not "
        "intended to replace broad language training. all control markers "
        "match the compact v13 runtime.\n",
        encoding="utf-8")

    for path, stats in ((fundamentals_path, fundamental_stats),
                        (curriculum_path, curriculum_stats)):
        print(
            f"wrote {path} · {stats['dialogues']:,} dialogues · "
            f"{stats['turns']:,} turns · {stats['bytes'] / 1024 / 1024:.1f} mb")
    print(f"wrote {manifest_path}")
    print(f"wrote {readme_path}")
    print(tokenizer_check)


if __name__ == "__main__":
    main()
