import importlib.util
import sys
import types
from pathlib import Path

# app has no __init__.py (namespace package); make the repo root importable so
# `import app.faster_whisper.utils` resolves when pytest is run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# app.faster_whisper.utils imports ctranslate2 at module load. Only when the
# native ML stack is genuinely NOT installed (e.g. a lightweight local run of
# the writer tests) do we stub it so the pure-logic tests can import the module.
# When ctranslate2 IS installed (the normal Poetry/CI env) we leave it untouched
# — find_spec detects installation without importing — so other tests exercise
# the real dependency rather than this fake.
if importlib.util.find_spec("ctranslate2") is None:
    ct2 = types.ModuleType("ctranslate2")
    converters = types.ModuleType("ctranslate2.converters")
    transformers = types.ModuleType("ctranslate2.converters.transformers")
    transformers.TransformersConverter = object
    converters.transformers = transformers
    ct2.converters = converters
    sys.modules["ctranslate2"] = ct2
    sys.modules["ctranslate2.converters"] = converters
    sys.modules["ctranslate2.converters.transformers"] = transformers
