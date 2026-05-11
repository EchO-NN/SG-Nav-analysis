import json
import re
from typing import List, Optional


ALLOWED_RELATIONS = [
    "on top of",
    "in front of",
    "opposite to",
    "next to",
    "close to",
    "left of",
    "right of",
    "inside",
    "behind",
    "above",
    "below",
    "under",
    "around",
    "beside",
    "near",
    "on",
]


def strip_thinking(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def extract_json(text: str):
    text = strip_thinking(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return None


def parse_probability_01(text: str, default: float = 0.0) -> float:
    text = strip_thinking(text)
    data = extract_json(text)

    if isinstance(data, dict):
        for key in ["probability", "score", "value", "p", "answer"]:
            if key in data:
                return parse_probability_01(str(data[key]), default=default)
    elif isinstance(data, (int, float)):
        return float(max(0.0, min(1.0, data)))

    text_no_percent = text.replace("%", "")
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text_no_percent)
    if not nums:
        return default

    value = float(nums[-1])
    if "%" in text:
        value = value / 100.0
    return float(max(0.0, min(1.0, value)))


def parse_distance_m(text: str, default: float = 10.0) -> float:
    text = strip_thinking(text)
    data = extract_json(text)

    if isinstance(data, dict):
        for key in ["distance", "distance_m", "meters", "metres", "value", "answer"]:
            if key in data:
                return parse_distance_m(str(data[key]), default=default)
    elif isinstance(data, (int, float)):
        return float(max(0.05, data))

    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
    if not nums:
        return float(default)

    value = float(nums[-1])
    lowered = text.lower()
    if "cm" in lowered and "m" not in lowered.replace("cm", ""):
        value = value / 100.0
    return float(max(0.05, value))


def parse_yes_no(text: str, default: bool = False) -> bool:
    text = strip_thinking(text).strip().lower()
    data = extract_json(text)
    if isinstance(data, dict):
        for key in ["answer", "result", "yes", "valid"]:
            if key in data:
                return parse_yes_no(str(data[key]), default=default)
    elif isinstance(data, bool):
        return data

    cleaned = re.sub(r"[^a-z ]", " ", text).strip()
    tokens = cleaned.split()
    if not tokens:
        return default

    first = tokens[0]
    if first in ["yes", "true", "valid", "correct"]:
        return True
    if first in ["no", "false", "invalid", "incorrect"]:
        return False

    if "yes" in tokens and "no" not in tokens:
        return True
    if "no" in tokens and "yes" not in tokens:
        return False
    return default


def parse_room_name(text: str, room_names: List[str]) -> Optional[str]:
    text = strip_thinking(text).lower()
    data = extract_json(text)
    if isinstance(data, dict):
        for key in ["room", "answer", "prediction"]:
            if key in data:
                return parse_room_name(str(data[key]), room_names)

    normalized_rooms = {room.lower(): room for room in room_names}
    aliases = {
        "living area": "living room",
        "lounge area": "lounge",
        "study": "office room",
        "office": "office room",
        "laundry": "laundry room",
        "dining area": "dining room",
    }
    for alias, canonical in aliases.items():
        if alias in text and canonical in normalized_rooms:
            return normalized_rooms[canonical]

    for room in room_names:
        if room.lower() in text:
            return room
    return None


def canonicalize_relation(text: str) -> str:
    text = strip_thinking(text).lower().strip()
    text = re.sub(r"^[\-\*\d\.\)\:\s]+", "", text)
    text = text.strip(" .;:,\"'")

    replacements = {
        "beside": "next to",
        "adjacent to": "next to",
        "nearby": "near",
        "close": "near",
        "on the top of": "on top of",
        "infront of": "in front of",
        "opposite": "opposite to",
    }
    if text in replacements:
        return replacements[text]

    for relation in sorted(ALLOWED_RELATIONS, key=len, reverse=True):
        if relation in text:
            return replacements.get(relation, relation)
    return text[:64] if text else "near"


def parse_relation_lines(text: str, expected_n: int) -> Optional[List[str]]:
    if expected_n <= 0:
        return []

    text = strip_thinking(text)
    data = extract_json(text)
    relations = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key in ["relationship", "relationships", "relation", "answer"]:
                    if key in item:
                        relations.append(canonicalize_relation(str(item[key])))
                        break
            elif isinstance(item, str):
                relations.append(canonicalize_relation(item))
    elif isinstance(data, dict):
        for key in ["relationships", "relations", "answers"]:
            if key in data:
                return parse_relation_lines(json.dumps(data[key]), expected_n)

    if not relations:
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith(
                ("here", "sure", "output", "response", "relationships")
            ):
                continue
            line = re.sub(r"^\s*[-*]\s*", "", line)
            line = re.sub(r"^\s*\d+[\).:\-]\s*", "", line)
            line = line.strip()
            if line:
                lines.append(canonicalize_relation(line))
        relations = lines

    if len(relations) >= expected_n:
        return relations[:expected_n]
    return None
