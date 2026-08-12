import sys
import types
from pathlib import Path

# app has no __init__.py (namespace package); make the repo root importable so
# `import app.faster_whisper.utils` resolves when pytest is run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# app.faster_whisper.utils imports ctranslate2 at module load. Stub it so the
# pure-logic unit tests below don't require the native ML stack (ctranslate2,
# torch, faster-whisper) to be installed.
if "ctranslate2" not in sys.modules:
    ct2 = types.ModuleType("ctranslate2")
    converters = types.ModuleType("ctranslate2.converters")
    transformers = types.ModuleType("ctranslate2.converters.transformers")
    transformers.TransformersConverter = object
    converters.transformers = transformers
    ct2.converters = converters
    sys.modules["ctranslate2"] = ct2
    sys.modules["ctranslate2.converters"] = converters
    sys.modules["ctranslate2.converters.transformers"] = transformers
