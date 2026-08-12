"""Unit tests for the faster-whisper default `json` output formatter (AMLG-13252).

Regression guard: `format_json` iterated `getattr(segment, "words", [])`, but
faster-whisper sets `segment.words = None` when word timestamps are not
requested — so the attribute exists and the `[]` default never applies, and
iterating `None` raised `TypeError`. `FakeSegment.words` defaults to None here,
so a regression re-introduces the crash.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from app.faster_whisper.utils import format_json


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float


@dataclass
class FakeSegment:
    start: float = 0.0
    end: float = 1.5
    text: str = "hello"
    tokens: List[int] = field(default_factory=lambda: [1, 2, 3])
    temperature: float = 0.0
    avg_logprob: float = -0.25
    compression_ratio: float = 1.1
    no_speech_prob: float = 0.01
    words: Optional[List[FakeWord]] = None  # None == word_timestamps not requested


def make_result(segment):
    return {"text": "hello", "language": "en", "segments": [segment]}


def test_format_json_without_word_timestamps_does_not_raise():
    # segment.words is None (the bug trigger).
    out = format_json(make_result(FakeSegment()))
    assert out["language"] == "en"
    assert out["text"] == "hello"
    assert len(out["segments"]) == 1
    assert out["segments"][0]["words"] == []
    assert out["segments"][0]["avg_logprob"] == -0.25


def test_format_json_with_word_timestamps():
    seg = FakeSegment(words=[FakeWord(start=0.0, end=0.5, word="hi", probability=0.9)])
    out = format_json(make_result(seg))
    assert out["segments"][0]["words"] == [
        {"word": "hi", "start": 0.0, "end": 0.5, "probability": 0.9}
    ]


def test_format_json_empty_segments():
    out = format_json({"text": "", "language": "en", "segments": []})
    assert out == {"text": "", "segments": [], "language": "en"}


@pytest.mark.parametrize("words_value", [None, []])
def test_format_json_empty_or_missing_words(words_value):
    out = format_json(make_result(FakeSegment(words=words_value)))
    assert out["segments"][0]["words"] == []
