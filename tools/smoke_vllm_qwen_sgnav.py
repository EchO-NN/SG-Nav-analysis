import argparse
import base64
import os
import sys
from io import BytesIO

import requests
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.llm_parsing import (
    parse_probability_01,
    parse_relation_lines,
    parse_yes_no,
    strip_thinking,
)


EDGE_PROMPT = """You are an indoor spatial relationship classifier.
For each object pair, output one short spatial relation.
Return a JSON array with exactly one object per input pair.
Each object must have exactly this key: "relationship".
Do not output markdown or explanation.

Input pairs:
[{"object1": "table", "object2": "chair"}, {"object1": "bed", "object2": "nightstand"}, {"object1": "lamp", "object2": "desk"}, {"object1": "sink", "object2": "mirror"}, {"object1": "sofa", "object2": "tv"}]
"""


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def chat(base_url, api_key, model, messages, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "seed": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"].get("content", "")


def assert_no_think(label, text):
    if "<think" in text.lower() or "</think>" in text.lower():
        print(f"[FAIL] {label} contains thinking output: {text!r}")
        raise SystemExit(1)


def has_number(text):
    import re

    return re.search(r"[-+]?\d*\.\d+|[-+]?\d+", strip_thinking(text)) is not None


def image_message(prompt, image):
    image_str = encode_image(image)
    return [
        {
            "role": "system",
            "content": "You are a strict visual classifier. Return only the requested short answer.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_str}"},
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "qwen3-vl-8b-instruct"))
    parser.add_argument("--image", default="segment_anything/notebooks/images/groceries.jpg")
    args = parser.parse_args()

    system = {
        "role": "system",
        "content": "Follow the requested output format exactly. Do not output hidden reasoning.",
    }

    probability_prompt = (
        "Return exactly one decimal number from 0 to 1.\n"
        "Question: What is the probability of a table and a chair appearing together?\n"
        "Answer:"
    )
    raw = chat(
        args.base_url,
        args.api_key,
        args.model,
        [system, {"role": "user", "content": probability_prompt}],
        max_tokens=16,
    )
    assert_no_think("text probability", raw)
    if not has_number(raw):
        print(f"[FAIL] probability parse failed raw={raw!r}")
        raise SystemExit(1)
    print(f"[OK] text probability raw={raw!r} parsed={parse_probability_01(raw):.3f}")

    raw = chat(
        args.base_url,
        args.api_key,
        args.model,
        [system, {"role": "user", "content": EDGE_PROMPT}],
        max_tokens=80,
    )
    assert_no_think("edge proposal", raw)
    relations = parse_relation_lines(raw, expected_n=5)
    if relations is None:
        print(f"[FAIL] edge proposal count mismatch raw={raw!r}")
        raise SystemExit(1)
    print(f"[OK] edge proposal raw={raw!r} parsed_count={len(relations)}")

    if not os.path.exists(args.image):
        print(f"[FAIL] image not found: {args.image}")
        raise SystemExit(1)

    image = Image.open(args.image).convert("RGB")
    raw = chat(
        args.base_url,
        args.api_key,
        args.model,
        image_message("Is there at least one object in the image? Return exactly yes or no.", image),
        max_tokens=8,
    )
    assert_no_think("image yes/no", raw)
    print(f"[OK] image yes/no raw={raw!r} parsed={parse_yes_no(raw)}")

    raw_relation = chat(
        args.base_url,
        args.api_key,
        args.model,
        image_message("Name one visible spatial relationship in the image. Return a short phrase.", image),
        max_tokens=16,
    )
    assert_no_think("image relation", raw_relation)
    if not raw_relation.strip():
        print("[FAIL] image relation response was empty")
        raise SystemExit(1)
    print(f"[OK] image relation raw={raw_relation!r}")

    red = Image.new("RGB", (96, 96), (255, 0, 0))
    blue = Image.new("RGB", (96, 96), (0, 0, 255))
    red_raw = chat(
        args.base_url,
        args.api_key,
        args.model,
        image_message("What is the dominant color? Return one word.", red),
        max_tokens=8,
    )
    blue_raw = chat(
        args.base_url,
        args.api_key,
        args.model,
        image_message("What is the dominant color? Return one word.", blue),
        max_tokens=8,
    )
    assert_no_think("red image color", red_raw)
    assert_no_think("blue image color", blue_raw)
    if strip_thinking(red_raw).lower() == strip_thinking(blue_raw).lower():
        print(f"[FAIL] image prompts look ignored red={red_raw!r} blue={blue_raw!r}")
        raise SystemExit(1)
    print(f"[OK] image transport red={red_raw!r} blue={blue_raw!r}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"[FAIL] vLLM request failed: {exc}")
        sys.exit(1)
