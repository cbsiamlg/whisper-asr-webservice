import pytest
import requests
import json
from google.cloud import storage
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://devwhisper.amlg.io"
BUCKET_NAME = "amlg-dev-playground"
TEST_VIDEO_NAME = "test_assets/beavis_and_butthead.mp4"


def get_test_video(convert_to_audio=True, trim=True):
    try:
        storage_client = storage.Client(project="i-amlg-dev")
        bucket = storage_client.get_bucket(BUCKET_NAME)
        blob = bucket.get_blob(TEST_VIDEO_NAME)
    except Exception as e:
        logger.error(
            f"Error accessing Google Cloud Storage: {str(e)}"
            "Make sure you've authenticated with `gcloud auth login` so the credentials are read."
        )

    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        blob.download_to_filename(temp_video_file.name)

        if convert_to_audio:
            temp_audio_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            )
            try:
                video = VideoFileClip(temp_video_file.name)
                audio = video.audio
                if trim:
                    audio = audio.subclip(0, 30)
                audio = audio.set_fps(16000)
                audio.write_audiofile(temp_audio_file.name, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
                video.close()

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
        video = VideoFileClip(video_path)
        trimmed_video = video.subclip(start_time, end_time)
        trimmed_video.write_videofile(temp_trimmed_file.name)
        video.close()
        trimmed_video.close()
        return temp_trimmed_file.name
    except Exception as e:
        logger.error(f"Error trimming video: {e}")
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

    logger.info(f"Transcription Status Code: {response.status_code}")
    logger.info(f"Transcription Headers: {response.headers}")
    logger.info(f"Transcription Content (sample): {response.text[:100]}")

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


def test_liveness():
    try:
        response = requests.get(f"{BASE_URL}/liveness/", allow_redirects=True)
        response.raise_for_status()
        assert response.json() == {"status": "ok"}
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Liveness check failed: {str(e)}")

def test_readiness():
    try:
        response = requests.get(f"{BASE_URL}/readiness/", allow_redirects=True)
        response.raise_for_status()
        assert response.json() == {"status": "ok"}
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Readiness check failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__])
