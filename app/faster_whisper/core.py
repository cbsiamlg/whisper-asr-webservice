import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch
import logging
import whisper
from .utils import (
    ResultWriter,
    WriteTXT,
    WriteSRT,
    WriteVTT,
    WriteTSV,
    WriteJSON,
    WriteRawJSON,
)
from faster_whisper import WhisperModel

# Load the pre-converted ctranslate2 model that faster-whisper hosts on the HF
# Hub (e.g. medium.en -> Systran/faster-whisper-medium.en). We used to convert
# openai/whisper-<model> ourselves via ctranslate2's TransformersConverter, but
# transformers 5.x (AMLG-13277) removed WhisperTokenizer.additional_special_tokens_ids
# which that converter relies on, breaking the conversion at cold start. Passing
# the model name lets faster-whisper fetch the already-converted model instead;
# AMLG-13280 pre-bakes it into the image at build time so no runtime download.
model_name = os.getenv("ASR_MODEL", "base")

if torch.cuda.is_available():
    model = WhisperModel(model_name, device="cuda", compute_type="float32")
else:
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
model_lock = Lock()
print(f"CUDA available: {torch.cuda.is_available()}")


def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
    word_timestamps: Union[bool, None],
    output,
):
    options_dict = {"task": task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    if word_timestamps:
        options_dict["word_timestamps"] = True
    with model_lock:
        segments = []
        text = ""
        i = 0
        segment_generator, info = model.transcribe(audio, beam_size=5, **options_dict)
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        result = {
            "language": options_dict.get("language", info.language),
            "segments": segments,
            "text": text,
        }

    outputFile = StringIO()
    write_result(result, outputFile, output)
    outputFile.seek(0)

    return outputFile


def language_detection(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.pad_or_trim(audio)

    # detect the spoken language
    with model_lock:
        segments, info = model.transcribe(audio, beam_size=5)
        detected_lang_code = info.language

    return detected_lang_code


def write_result(result: dict, file: BinaryIO, output: Union[str, None]):
    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(result, file=file)
    elif output == "raw_json":
        WriteRawJSON(ResultWriter).write_result(result, file=file)
    elif output == "txt":
        WriteTXT(ResultWriter).write_result(result, file=file)
    else:
        return "Please select an output method!"
