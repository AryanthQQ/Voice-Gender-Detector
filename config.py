"""
config.py — Load .env settings for Voice Gender App
"""
import os

def _load_env(path='.env'):
    """Simple .env file parser — no external deps needed."""
    env = {}
    env_path = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, _, val = line.partition('=')
                    env[key.strip()] = val.strip()
    return env

_cfg = _load_env()

# Push ALL .env values into os.environ so other modules (deepfake_detector, etc.) can read them
for _k, _v in _cfg.items():
    os.environ.setdefault(_k, _v)

# SMTP Email Config (Optional)
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
EMAIL_FROM = os.environ.get("EMAIL_FROM", SMTP_USERNAME)
EMAIL_TO = os.environ.get("EMAIL_TO")



# When to notify: 'all' = every upload, 'female' = only female detections
NOTIFY_ON: str = _cfg.get('NOTIFY_ON', 'all')

# Shared secret required (via X-API-Key header) to call /predict-url and /api/admin/* routes
API_KEY = os.environ.get("API_KEY", "")

# Max audio jobs (STT + deepfake + gender) processed at once. Tune based on
# available RAM/CPU — each job loads Whisper + ensemble models into memory.
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", 2))

# Public URL where this server is reachable — used to build the admin dashboard
# link in manual_review email alerts. Set to your real domain in production.
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:8000")

# Audio retention: auto-decided requests (clean accept/reject) are deleted right
# after processing — no audio is kept. Only manual_review audio is kept, and only
# for this many days, so a human has time to review it before it's purged.
MANUAL_REVIEW_RETENTION_DAYS = int(os.environ.get("MANUAL_REVIEW_RETENTION_DAYS", 7))

def email_configured() -> bool:
    """Returns True if Email credentials are set and not placeholder."""
    return (
        bool(SMTP_USERNAME)
        and SMTP_USERNAME != 'YOUR_EMAIL@gmail.com'
        and bool(SMTP_PASSWORD)
        and SMTP_PASSWORD != 'YOUR_APP_PASSWORD'
        and bool(EMAIL_TO)
    )


import platform

# Base Storage Directory
# Windows defaults to ./data, Linux defaults to /data/voiceguard
_os_name = platform.system().lower()
_default_base = "./data" if _os_name == "windows" else "/data/voiceguard"

STORAGE_BASE = os.environ.get("STORAGE_BASE", _default_base)

RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", os.path.join(STORAGE_BASE, "recordings"))
TEMP_UPLOADS_DIR = os.environ.get("TEMP_UPLOADS_DIR", os.path.join(STORAGE_BASE, "uploads"))
LOGS_DIR = os.environ.get("LOGS_DIR", os.path.join(STORAGE_BASE, "logs"))
DATASET_DIR = os.environ.get("DATASET_DIR", os.path.join(STORAGE_BASE, "dataset"))
FAILED_DIR = os.environ.get("FAILED_DIR", os.path.join(STORAGE_BASE, "failed"))
MANUAL_REVIEW_DIR = os.environ.get("MANUAL_REVIEW_DIR", os.path.join(STORAGE_BASE, "manual_review"))
BACKUPS_DIR = os.environ.get("BACKUPS_DIR", os.path.join(STORAGE_BASE, "backups"))

# Auto-create all directories safely
for _dir in [RECORDINGS_DIR, TEMP_UPLOADS_DIR, LOGS_DIR, DATASET_DIR, FAILED_DIR, MANUAL_REVIEW_DIR, BACKUPS_DIR]:
    try:
        os.makedirs(_dir, exist_ok=True)
    except Exception as e:
        print(f"[WARN] Could not create directory {_dir}: {e}")
