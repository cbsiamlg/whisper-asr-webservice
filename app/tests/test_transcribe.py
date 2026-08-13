import logging

import pytest
import requests
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://devwhisper.amlg.io"

# Durable ASR fixture (AMLG-13281). The previous fixture lived in the now-deleted
# gs://amlg-dev-playground bucket and was a video-only MP4 that had to be
# pre-converted to WAV with moviepy. This curated asset carries a real audio
# track, so the service's ffmpeg pipe decodes it directly and the test can POST
# the bytes as-is — no moviepy pre-conversion.
BUCKET_NAME = "amlg-dev"
TEST_AUDIO_NAME = "ASR-test/audio-files/one_min-test-456.wav"
GCP_PROJECT = "i-amlg-dev"


def get_test_audio():
    """Download the curated audio fixture bytes from GCS.

    Requires application-default credentials — run `gcloud auth login` (or set
    GOOGLE_APPLICATION_CREDENTIALS) before invoking this integration test.
    """
    try:
        storage_client = storage.Client(project=GCP_PROJECT)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(TEST_AUDIO_NAME)
        return blob.download_as_bytes()
    except Exception as e:
        pytest.fail(
            f"Error accessing gs://{BUCKET_NAME}/{TEST_AUDIO_NAME}: {e}. "
            "Make sure you've authenticated with `gcloud auth login` so the "
            "credentials are read."
        )


def test_transcription_request(word_timestamps=True):
    audio_content = get_test_audio()
    files = {"audio_file": audio_content}
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
    audio_content = get_test_audio()
    files = {"audio_file": audio_content}
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
