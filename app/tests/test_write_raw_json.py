"""Unit tests for the faster-whisper `raw_json` output writer (AMLG-13219).

Regression guard: `WriteRawJSON.format_segments` used to index the faster-whisper
`Segment` positionally (`segment[2]`...), but `Segment` is a dataclass and is not
subscriptable, so `output=raw_json` raised `TypeError`. The fix uses attribute
access. `FakeSegment` below is a dataclass (like the real one), so any regression
to positional indexing fails these tests with `TypeError`.
"""
import io
import json
from dataclasses import dataclass
from typing import List, Optional

import pytest

from app.faster_whisper.utils import WriteRawJSON


@dataclass
class FakeSegment:
    # Field order mirrors faster_whisper.transcribe.Segment; the essential
    # property is that this is a dataclass (not subscriptable).
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[list]
    temperature: Optional[float]


def make_segment(**overrides):
    defaults = dict(
        id=1,
        seek=0,
        start=0.0,
        end=1.5,
        text="hello",
        tokens=[1, 2, 3],
        avg_logprob=-0.25,
        compression_ratio=1.1,
        no_speech_prob=0.01,
        words=None,
        temperature=0.0,
    )
    defaults.update(overrides)
    return FakeSegment(**defaults)


@pytest.fixture
def writer():
    # output_dir is only used by ResultWriter.__call__, not by format_segments /
    # write_result, so None is fine for these tests.
    return WriteRawJSON(output_dir=None)


def test_format_segments_maps_fields_by_name(writer):
    result = writer.format_segments({"segments": [make_segment()]})

    assert result == [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": 1.5,
            "text": "hello",
            "tokens": [1, 2, 3],
            "temperature": 0.0,
            "avg_logprob": -0.25,
            "compression_ratio": 1.1,
            "no_speech_prob": 0.01,
        }
    ]


def test_format_segments_empty(writer):
    assert writer.format_segments({"segments": []}) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", None),
        ("no_speech_prob", 0.0),
        ("no_speech_prob", 1.0),
        ("tokens", []),
        ("text", ""),
    ],
)
def test_format_segments_edge_values_pass_through(writer, field, value):
    result = writer.format_segments({"segments": [make_segment(**{field: value})]})
    assert result[0][field] == value


def test_write_result_emits_valid_json(writer):
    buf = io.StringIO()
    writer.write_result(
        {"text": "hello", "language": "en", "segments": [make_segment()]}, buf
    )
    buf.seek(0)

    data = json.loads(buf.getvalue())
    assert data["text"] == "hello"
    assert data["language"] == "en"
    assert data["segments"][0]["avg_logprob"] == -0.25
    assert data["segments"][0]["no_speech_prob"] == 0.01
