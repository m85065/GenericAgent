import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTENDS = REPO_ROOT / "frontends"
for path in (str(REPO_ROOT), str(FRONTENDS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from chatapp_common import completed_sentence_prefix


def test_sentence_prefix_requires_sentence_end():
    assert completed_sentence_prefix("hello world") == ""


def test_sentence_prefix_returns_completed_sentence_only():
    assert completed_sentence_prefix("hello world. next") == "hello world."


def test_sentence_prefix_supports_chinese_punctuation():
    assert completed_sentence_prefix("第一句。第二句") == "第一句。"


def test_sentence_prefix_keeps_decimal_point_unfinished():
    assert completed_sentence_prefix("value is 3.14 now") == ""


def test_sentence_prefix_uses_newline_as_boundary():
    assert completed_sentence_prefix("line1\nline2") == "line1"


def test_sentence_prefix_handles_punctuation_before_newline():
    assert completed_sentence_prefix("sentence.\nmore text") == "sentence."
