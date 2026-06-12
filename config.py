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

# Telegram settings
TELEGRAM_BOT_TOKEN: str = _cfg.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID: str   = _cfg.get('TELEGRAM_CHAT_ID', '')

# Recordings directory (relative to main.py location)
RECORDINGS_DIR: str = _cfg.get('RECORDINGS_DIR', 'recordings')

# When to notify: 'all' = every upload, 'female' = only female detections
NOTIFY_ON: str = _cfg.get('NOTIFY_ON', 'all')

def telegram_configured() -> bool:
    """Returns True if Telegram credentials are set and not placeholder."""
    return (
        bool(TELEGRAM_BOT_TOKEN)
        and TELEGRAM_BOT_TOKEN != 'YOUR_BOT_TOKEN_HERE'
        and bool(TELEGRAM_CHAT_ID)
        and TELEGRAM_CHAT_ID != 'YOUR_CHAT_ID_HERE'
    )

