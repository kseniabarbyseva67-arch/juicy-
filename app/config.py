from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv('DATA_DIR', BASE_DIR / 'data'))
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv('DB_PATH', DATA_DIR / 'gallery.sqlite3'))
DEFAULT_TOP_K = int(os.getenv('TOP_K', '10'))
REFUSAL_THRESHOLD = float(os.getenv('REFUSAL_THRESHOLD', '0.82'))
MAX_IMAGE_MB = int(os.getenv('MAX_IMAGE_MB', '20'))
EMBEDDING_DIM = 512
