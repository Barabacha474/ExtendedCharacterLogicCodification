from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"
DATA_SPLITS = DATA / "splits"

RESULTS = ROOT / "results"
RESULTS_PREDICTIONS = RESULTS / "predictions"
RESULTS_METRICS = RESULTS / "metrics"
RESULTS_LOGS = RESULTS / "logs"

MODELS = ROOT / "models"
VENDOR = ROOT / "vendor"

for _p in (DATA_PROCESSED, DATA_SPLITS, RESULTS_PREDICTIONS, RESULTS_METRICS, RESULTS_LOGS):
    _p.mkdir(parents=True, exist_ok=True)