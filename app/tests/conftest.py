import sys
from pathlib import Path

# app has no __init__.py (namespace package); make the repo root importable so
# `import app.faster_whisper.utils` resolves when pytest is run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOTE: app.faster_whisper.utils used to import ctranslate2 at module load (for
# the model converter), which required stubbing ctranslate2 for lightweight
# runs of the pure-logic writer tests. The converter was removed in AMLG-13277
# (transformers 5.x broke it; faster-whisper now loads a pre-converted model),
# so utils.py no longer imports ctranslate2 and no stub is needed.
