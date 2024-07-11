import pytest
import requests
import json
from google.cloud import storage
import ffmpeg
import tempfile
import os

BASE_URL = "https://devwhisper.amlg.io"
BUCKET_NAME = "amlg-dev-playground"
TEST_VIDEO_NAME = "test_assets/beavis_and_butthead.mp4"


def get_test_video(convert_to_audio=True, trim=True):
    storage_client = storage.Client(project="i-amlg-dev")
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(TEST_VIDEO_NAME)

    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        blob.download_to_filename(temp_video_file.name)

        if convert_to_audio:
            temp_audio_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )
            try:
                stream = ffmpeg.input(temp_video_file.name)
                audio = stream.audio
                if trim:
                    audio = audio.filter("atrim", start=0, duration=30)
                output = ffmpeg.output(
                    audio,
                    temp_audio_file.name,
                    acodec="pcm_s16le",
                    ac=1,
                    ar="16k"
                )

                try:
                    ffmpeg.run(
                        output,
                        overwrite_output=True,
                        capture_stderr=True
                    )
                except ffmpeg.Error as e:
                    print("ffmpeg stderr:", e.stderr.decode("utf8"))
                    raise

                with open(temp_audio_file.name, "rb") as f:
                    content = f.read()
            finally:
                os.unlink(temp_audio_file.name)
        else:
            content = blob.download_as_bytes()
    finally:
        os.unlink(temp_video_file.name)

    return content


def trim_video(video_path, start_time=0, end_time=120):
    temp_trimmed_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        input_stream = ffmpeg.input(video_path)
        output = ffmpeg.output(
            input_stream.trim(
                start=start_time,
                end=end_time
            ),
            temp_trimmed_file.name
        )
        ffmpeg.run(output, overwrite_output=True)
        return temp_trimmed_file.name
    except Exception as e:
        print(f"Error trimming video: {e}")
        os.unlink(temp_trimmed_file.name)
        raise


def test_transcription_request(short_video=True, word_timestamps=True):
    video_content = get_test_video(convert_to_audio=True, trim=short_video)
    files = {"audio_file": video_content}
    params = {
        "task": "transcribe",
        "language": "en",
        "encode": True,
        "output": "json",
        "word_timestamps": word_timestamps,
    }
    response = requests.post(url=f"{BASE_URL}/asr", files=files, params=params)

    print(f"Transcription Status Code: {response.status_code}")
    print(f"Transcription Headers: {response.headers}")
    print(f"Transcription Content (sample): {response.text[:100]}")

    assert response.status_code == 200, "Transcription request failed"
    assert len(response.text) > 0, "Transcription returned empty result"


def test_language_detection():
    video_content = get_test_video()
    files = {"audio_file": video_content}
    response = requests.post(f"{BASE_URL}/detect-language", files=files)
    assert response.status_code == 200, "Language detection request failed"
    result = response.json()
    assert (
        "detected_language" in result
    ), "Language detection result is missing 'detected_language'"
    assert (
        "language_code" in result
    ), "Language detection result is missing 'language_code'"


if __name__ == "__main__":
    pytest.main([__file__])
