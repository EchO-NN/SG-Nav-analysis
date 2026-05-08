import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.llm_parsing import (
    canonicalize_relation,
    parse_probability_01,
    parse_relation_lines,
    parse_room_name,
    parse_yes_no,
    strip_thinking,
)


def assert_equal(actual, expected):
    if actual != expected:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def assert_close(actual, expected, eps=1e-6):
    if abs(actual - expected) > eps:
        raise AssertionError(f"expected {expected!r}, got {actual!r}")


def main():
    assert_equal(strip_thinking("<think>hidden</think>\n0.72"), "0.72")
    assert_close(parse_probability_01("The probability is 0.72."), 0.72)
    assert_close(parse_probability_01('{"probability": 0.31}'), 0.31)
    assert_close(parse_probability_01("final answer: 64%"), 0.64)

    assert_equal(parse_yes_no("No, although it might seem yes."), False)
    assert_equal(parse_yes_no('{"answer": "yes"}'), True)
    assert_equal(parse_yes_no("The answer is yes."), True)

    rooms = ["bedroom", "living room", "bathroom", "office room"]
    assert_equal(parse_room_name('{"room": "living room"}', rooms), "living room")
    assert_equal(parse_room_name("I would choose the study.", rooms), "office room")
    assert_equal(parse_room_name("living area", rooms), "living room")

    assert_equal(canonicalize_relation("1. On the top of."), "on top of")
    assert_equal(canonicalize_relation("adjacent to"), "next to")
    assert_equal(
        parse_relation_lines(
            '[{"relationship": "next to"}, {"relationship": "opposite"}]',
            2,
        ),
        ["next to", "opposite to"],
    )
    assert_equal(
        parse_relation_lines("1. on top of\n2. left of\n3. near", 2),
        ["on top of", "left of"],
    )
    assert_equal(parse_relation_lines("next to", 2), None)
    print("[OK] llm parsing tests passed")


if __name__ == "__main__":
    main()
