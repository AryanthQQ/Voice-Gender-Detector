"""
Voice Gender Detection - FastAPI Backend
- Accepts audio files (WAV/MP3/OGG etc.)
- Extracts acoustic features via librosa/soundfile (used for the pitch
  safety filter and the UI's frequency display)
- Runs a pretrained Wav2Vec2-XLSR model as the primary gender classifier
- Auto-saves every recording to recordings/ folder
- Sends Email notification to admin with verification result
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import imageio_ffmpeg
import tempfile
import threading
import urllib.request
from datetime import datetime
import soundfile as sf
import ssl
import logging
from logging.handlers import RotatingFileHandler
import uuid
import time
import config



_base_logger = logging.getLogger("voice_gender_api")
_base_logger.setLevel(logging.INFO)
_base_logger.propagate = False
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

info_handler = RotatingFileHandler(os.path.join(config.LOGS_DIR, "application.log"), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = RotatingFileHandler(os.path.join(config.LOGS_DIR, "error.log"), maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

_base_logger.addHandler(info_handler)
_base_logger.addHandler(error_handler)

class SafeLogger:
    def __init__(self, logger):
        self._logger = logger
    def info(self, msg, *args, **kwargs):
        try: self._logger.info(msg, *args, **kwargs)
        except Exception: pass
    def warning(self, msg, *args, **kwargs):
        try: self._logger.warning(msg, *args, **kwargs)
        except Exception: pass
    def exception(self, msg, *args, **kwargs):
        try: self._logger.exception(msg, *args, **kwargs)
        except Exception: pass

logger = SafeLogger(_base_logger)

def safe_load_audio(path, sr=16000):
    import soundfile as sf
    import librosa
    import subprocess
    import tempfile
    import os
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Try soundfile first for standard wav files (fast path)
    if path.lower().endswith(('.wav', '.flac', '.ogg')):
        try:
            y, orig_sr = sf.read(path, dtype='float32', always_2d=False)
            if y.ndim > 1: y = y.mean(axis=1)
            if orig_sr != sr: y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
            return y, sr
        except Exception:
            pass # Fallback to ffmpeg
            
    # Fallback to ffmpeg for mp3, webm, m4a, or corrupted files
    fd, temp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        # Run ffmpeg to convert to 16kHz mono WAV safely
        cmd = [ffmpeg_exe, "-y", "-i", path, "-ar", str(sr), "-ac", "1", temp_wav]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise ValueError(f"Corrupted or unsupported audio format.")
        
        y, orig_sr = sf.read(temp_wav, dtype='float32', always_2d=False)
        return y, sr
    finally:
        if os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
            except:
                pass




# Ensure ffmpeg is available for audioread
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

import hmac
import ipaddress
import socket
from urllib.parse import urlparse

import numpy as np
import librosa

# librosa.pyin() JIT-compiles its Viterbi decoder via numba on first call. On Windows,
# that compilation crashes the process (segfault, exception 0xc0000005) if torch has
# already initialized its bundled OpenMP runtime beforehand — confirmed by reproduction.
# Compiling it here, before torch/transformers are imported below, avoids the conflict.
logger.info("[WARMUP] Pre-compiling librosa.pyin (numba JIT) before loading torch...")
librosa.pyin(np.zeros(16000 * 2, dtype=np.float32), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=16000)
logger.info("[WARMUP] Done.")

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool

from pydantic import BaseModel
import config

from deepfake_detector_v2 import AdvancedDeepfakeDetector
import gender_guesser.detector as gender
import fingerprint
import fingerprint_store

# Limit entire request pipeline to N concurrent tasks to prevent RAM exhaustion (tune via MAX_CONCURRENT_JOBS)
import threading
GLOBAL_PROCESS_LOCK = threading.Semaphore(config.MAX_CONCURRENT_JOBS)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Gender Detection API", version="2.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# Initialize name gender detector
gender_detector = gender.Detector()

# Initialize Advanced Deepfake Detector
advanced_deepfake_detector = AdvancedDeepfakeDetector()

fingerprint_store.init_db()

# Initialize Speech-to-Text Model
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
STT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"[STT] Loading Speech-to-Text model (Multi-lingual Whisper) on {STT_DEVICE}...")
stt_processor = WhisperProcessor.from_pretrained('openai/whisper-tiny')
stt_model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny').to(STT_DEVICE)
stt_model.eval()
logger.info("[STT] Speech-to-Text model loaded successfully!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(x_api_key: str = Header(default=None)):
    """Guards /predict-url and /api/admin/* — caller must send a matching X-API-Key header."""
    if not config.API_KEY:
        raise HTTPException(status_code=503, detail="Server misconfigured: API_KEY not set.")
    if not x_api_key or not hmac.compare_digest(x_api_key, config.API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


def _assert_public_url(url: str):
    """Blocks SSRF: only allow http/https URLs that resolve to a public IP address."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http:// and https:// URLs are allowed.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL is missing a hostname.")
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError(f"Could not resolve host: {hostname}")
    for family, _, _, _, sockaddr in addrs:
        ip = ipaddress.ip_address(sockaddr[0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified):
            raise ValueError(f"Refusing to fetch from non-public address: {ip}")

# ── Primary Gender Model ─────────────────────────────────────────────────────
import gender_verifier

try:
    gender_verifier.load_model()
    logger.info("[OK] Primary gender model (Wav2Vec2-XLSR) loaded successfully!")
except Exception as e:
    logger.exception(f"[ERR] Error loading primary gender model: {e}")

# ── Recordings directory ──────────────────────────────────────────────────────
RECORDINGS_DIR = config.RECORDINGS_DIR
TEMP_UPLOADS_DIR = config.TEMP_UPLOADS_DIR
FAILED_DIR = config.FAILED_DIR
MANUAL_REVIEW_DIR = config.MANUAL_REVIEW_DIR
logger.info(f"[OK] Recordings will be saved to: {RECORDINGS_DIR}")
logger.info(f"[OK] Temp uploads will be saved to: {TEMP_UPLOADS_DIR}")
logger.info(f"[OK] Failed recordings will be saved to: {FAILED_DIR}")
logger.info(f"[OK] Manual-review audio kept for {config.MANUAL_REVIEW_RETENTION_DAYS} days in: {MANUAL_REVIEW_DIR}")


def _delete_audio(path: str):
    """Deletes an audio file and its wav2vec embedding cache (path + '.npy'), if present."""
    for p in (path, path + ".npy"):
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception as e:
            logger.exception(f"[CLEANUP] Failed to delete {p}: {e}")


def _keep_for_manual_review(path: str) -> str:
    """Moves audio into MANUAL_REVIEW_DIR so a human can review it, subject to
    MANUAL_REVIEW_RETENTION_DAYS auto-purge. Returns the new path."""
    dest = os.path.join(MANUAL_REVIEW_DIR, os.path.basename(path))
    try:
        os.replace(path, dest)
    except Exception as e:
        logger.exception(f"[MANUAL REVIEW] Failed to move {path} to {dest}: {e}")
        return path
    # Move the embedding cache too, if any, so nothing audio-derived lingers elsewhere.
    try:
        if os.path.exists(path + ".npy"):
            os.replace(path + ".npy", dest + ".npy")
    except Exception:
        pass
    return dest


def _purge_expired_manual_review():
    """Background loop: deletes manual_review audio older than the retention window.
    This is the only place audio is meant to persist, and only temporarily."""
    max_age_s = config.MANUAL_REVIEW_RETENTION_DAYS * 86400
    while True:
        try:
            now = time.time()
            for fname in os.listdir(MANUAL_REVIEW_DIR):
                if fname.endswith(".npy"):
                    continue
                fpath = os.path.join(MANUAL_REVIEW_DIR, fname)
                try:
                    if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > max_age_s:
                        _delete_audio(fpath)
                        logger.info(f"[CLEANUP] Purged expired manual_review file: {fname}")
                except Exception:
                    pass
        except Exception as e:
            logger.exception(f"[CLEANUP] manual_review purge loop error: {e}")
        time.sleep(3600)  # check hourly


threading.Thread(target=_purge_expired_manual_review, daemon=True).start()

# ── Simple In-Memory Cache for n8n Loop Protection ────────────────────────────
processed_cache = {}

def _add_to_cache(url: str, result_dict: dict):
    processed_cache[url] = result_dict
    if len(processed_cache) > 1000:
        processed_cache.pop(next(iter(processed_cache)))

# ── Email Notifier ────────────────────────────────────────────────────────────
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.audio import MIMEAudio

class EmailNotifier:
    """Send Email messages with audio attachment using smtplib."""

    def __init__(self, server, port, username, password, from_addr, to_addr):
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addr = to_addr

    def send(self, subject: str, html_body: str, audio_path: str = None) -> bool:
        """Send an email. Returns True on success."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = self.to_addr
            msg['Subject'] = subject

            msg.attach(MIMEText(html_body, 'html'))

            if audio_path and os.path.exists(audio_path):
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                audio_part = MIMEAudio(audio_data, _subtype="wav")
                audio_part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(audio_path)}"')
                msg.attach(audio_part)

            server = smtplib.SMTP(self.server, self.port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()
            return True
        except Exception as ex:
            logger.exception(f"[EMAIL] Send failed: {ex}")
            return False


_notifier = None
if config.email_configured():
    _notifier = EmailNotifier(
        config.SMTP_SERVER, config.SMTP_PORT, 
        config.SMTP_USERNAME, config.SMTP_PASSWORD, 
        config.EMAIL_FROM, config.EMAIL_TO
    )
    logger.info("[OK] Email notifier ready.")
else:
    logger.info("[WARN] Email not configured. Edit .env to add SMTP credentials.")


def _build_email_message(result: dict, filename: str, file_size_kb: float, source_url: str = None) -> tuple:
    """Build a rich HTML email notification message. Returns (subject, html_body)"""
    ens    = result['ensemble']
    svm    = result['svm']
    gbm    = result['gbm']
    rf     = result['rf']
    label  = ens['label']
    conf   = ens['confidence']
    votes  = ens['male_votes']
    now    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    req_id = result.get('request_id', 'Unknown')
    adv_id = result.get('advisor_id', 'Unknown')
    reason = result.get('reason', 'Uncertain')
    audio_link = source_url if source_url else '#'

    # Manual Review Branch
    if label == 'manual_review' or result.get('status') == 'manual_review':
        subject = "⚠️ VoiceGuard Manual Review Required"
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
            <h2>VoiceGuard Manual Review</h2>
            
            <h3>Request Information</h3>
            <ul>
                <li><b>Request ID:</b> {req_id}</li>
                <li><b>Advisor ID:</b> {adv_id}</li>
                <li><b>Detection Status:</b> Manual Review</li>
                <li><b>Confidence:</b> {conf:.1f}%</li>
                <li><b>Detection Reason:</b> {reason}</li>
                <li><b>Processing Time:</b> {result.get('processing_time_ms', 0):.1f} ms</li>
                <li><b>Timestamp:</b> {now}</li>
            </ul>
            
            <h3>Audio</h3>
            <p>S3 Audio URL: <a href="{audio_link}" target="_blank">Open Recording</a></p>
            
            <hr style="margin-top: 20px; border: 0; border-top: 1px solid #eee;">
            <p style="font-size: 0.9em; color: #666;">
                This email was automatically generated by VoiceGuard AI.<br>
                Please review this recording manually.
            </p>
        </body>
        </html>
        """
        return subject, html_body

    # Default Branch for automatic decisions (Male/Female)
    if label == 'female':
        verdict_line = "VERDICT: FEMALE VERIFIED"
        status_emoji = "✅"
    else:
        verdict_line = "VERDICT: MALE DETECTED"
        status_emoji = "🔵"

    subject = f"{status_emoji} Voice Gender Alert: {label.upper()} ({conf:.1f}%)"

    female_votes = 3 - votes
    if label == 'female':
        vote_summary = f"{female_votes}/3 Female"
    else:
        vote_summary = f"{votes}/3 Male"

    dashboard_link = f"<p>🔗 <b>Action Required:</b> <a href=\"{config.PUBLIC_BASE_URL}/static/admin.html\">Go to Admin Dashboard to Review</a></p>"

    html_body = f"""
    <html><body>
        <h2>🎙️ Voice Gender Verification Alert</h2>
        <hr>
        <p>📅 <b>Time:</b> {now}<br>
        🆔 <b>Request ID:</b> {req_id}<br>
        👤 <b>Advisor ID:</b> {adv_id}<br>
        🔊 <b>File Name:</b> <code>{filename}</code></p>
        {dashboard_link}
        <h3>{status_emoji} {verdict_line}</h3>
        <p><b>Confidence:</b> {conf:.1f}%</p>
        <p><b>Reason:</b> {reason}</p>
        <hr>
        <h4>Model Breakdown:</h4>
        <ul>
            <li><b>AI Check:</b> {'🔴 Replay Attack' if 'Replay' in result.get('ai', {}).get('reason', '') else '🔴 AI/Deepfake' if result.get('ai', {}).get('is_ai') else '✅ Real Human'}</li>
            <li><b>SVM:</b> {svm['label'].title()} ({svm['confidence']:.0f}%)</li>
            <li><b>Gradient Boost:</b> {gbm['label'].title()} ({gbm['confidence']:.0f}%)</li>
            <li><b>Random Forest:</b> {rf['label'].title()} ({rf['confidence']:.0f}%)</li>
            <li><b>Ensemble Vote:</b> {vote_summary}</li>
        </ul>
        <hr>
        <p><i>Auto-verified by Voice Gender AI v2.0</i></p>
    </body></html>
    """
    return subject, html_body


def _notify_async(result: dict, filename: str, file_size_kb: float, saved_path: str = None):
    """Send Email notification in background thread (non-blocking)."""
    if _notifier is None:
        return
    label = result['ensemble']['label']
    if label != 'manual_review' and result.get('status') != 'manual_review':
        return  # Only send notifications for manual review cases
    # Extract source URL if available (from /predict-url flow)
    source_url = result.get('source_url') or None
    subject, html_body = _build_email_message(result, filename, file_size_kb, source_url=source_url)
    ok = _notifier.send(subject, html_body, audio_path=None) # Audio attachment disabled per requirement
    logger.info(f"[EMAIL] Notification {'sent' if ok else 'FAILED'} for {filename} ({label})")


def _dispatch_email_notification(result: dict, filename: str, file_size_kb: float, saved_path: str = None):
    """Helper to spawn the email notification background thread."""
    t = threading.Thread(
        target=_notify_async,
        args=(result, filename, file_size_kb, saved_path),
        daemon=True
    )
    t.start()


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(audio_path: str) -> dict:
    """Extract 20 acoustic features matching the training dataset."""

    try:
        y_raw, sr = sf.read(audio_path, dtype='float32', always_2d=False)
        if y_raw.ndim > 1:
            y_raw = y_raw.mean(axis=1)
        if sr != 16000:
            y = librosa.resample(y_raw, orig_sr=sr, target_sr=16000)
            sr = 16000
        else:
            y = y_raw
    except Exception as load_err:
        logger.info(f"[WARN] soundfile load failed ({load_err}), trying librosa fallback...")
        y, sr = safe_load_audio(audio_path, sr=16000)

    # ── AUDIO FILTERING (IMPROVES ACCURACY) ──────────────────────────────────
    # 1. Raw Silence / Blank Noise Detection (Check BEFORE normalization)
    if np.max(np.abs(y)) < 0.05:
        raise ValueError("Audio volume is very low or completely silent. Please speak loudly and clearly.")
        
    # 2. Silence Trimming (Shuru aur aakhir ka blank noise/shanti hatana)
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # 3. Short Audio Rejection (Enforce 4 second rule)
    if len(y) < 16000 * 4.0:
        raise ValueError("Audio is too short. Please speak for at least 4 seconds.")

    # 4. Volume Normalization (Aawaaz ka level barabar karna)
    y = librosa.util.normalize(y)

    # 3. Bandpass Filter (50Hz - 3000Hz) - Safe Noise Reduction
    import scipy.signal
    nyq = 0.5 * sr
    b, a = scipy.signal.butter(3, [50.0 / nyq, 3000.0 / nyq], btype='band')
    y = scipy.signal.filtfilt(b, a, y)
    # ─────────────────────────────────────────────────────────────────────────

    fmin, fmax = 50, 280

    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    mask = (freqs >= 0) & (freqs <= fmax)
    stft_filt = stft[mask, :]
    freqs_filt = freqs[mask]

    power = np.mean(stft_filt, axis=1)
    power_norm = power / (power.sum() + 1e-10)

    meanfreq = np.sum(freqs_filt * power_norm) / 1000.0
    sd = np.sqrt(np.sum(power_norm * (freqs_filt / 1000.0 - meanfreq) ** 2))

    cumpower = np.cumsum(power_norm)
    Q25    = freqs_filt[np.searchsorted(cumpower, 0.25)] / 1000.0
    median = freqs_filt[np.searchsorted(cumpower, 0.50)] / 1000.0
    Q75    = freqs_filt[np.searchsorted(cumpower, 0.75)] / 1000.0
    IQR    = Q75 - Q25

    skew = np.sum(power_norm * ((freqs_filt / 1000.0 - meanfreq) / (sd + 1e-10)) ** 3)
    kurt = np.sum(power_norm * ((freqs_filt / 1000.0 - meanfreq) / (sd + 1e-10)) ** 4)
    sp_ent = -np.sum(power_norm * np.log2(power_norm + 1e-10))
    sfm_val = librosa.feature.spectral_flatness(y=y)[0].mean()
    if sfm_val > 0.08:
        raise ValueError(f"Too much background noise detected (Noise level: {sfm_val:.2f}). Please record in a quieter environment.")
        
    mode_idx = np.argmax(power)
    mode = freqs_filt[mode_idx] / 1000.0
    centroid = np.sum(freqs_filt * power_norm) / 1000.0

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr
        )
        
        if voiced_flag is not None:
            voiced_ratio = np.sum(voiced_flag) / len(voiced_flag)
            if voiced_ratio < 0.25:
                raise ValueError(f"Voice is too faint compared to background noise. Please speak closer to the microphone.")
                
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        f0_voiced = np.array([])

    if len(f0_voiced) > 0:
        meanfun = np.mean(f0_voiced) / 1000.0
        minfun  = np.min(f0_voiced) / 1000.0
        maxfun  = np.max(f0_voiced) / 1000.0
    else:
        meanfun = 0.14
        minfun  = 0.08
        maxfun  = 0.20

    dom_freqs = []
    hop, frame_len = 512, 2048
    for i in range(0, len(y) - frame_len, hop):
        frame = y[i:i+frame_len]
        fft_frame = np.abs(np.fft.rfft(frame))
        fft_freqs = np.fft.rfftfreq(frame_len, 1.0 / sr)
        valid = (fft_freqs >= fmin) & (fft_freqs <= fmax)
        if valid.any():
            dom_freqs.append(fft_freqs[valid][np.argmax(fft_frame[valid])])

    if dom_freqs:
        dom_arr = np.array(dom_freqs) / 1000.0
        meandom = np.mean(dom_arr)
        mindom  = np.min(dom_arr)
        maxdom  = np.max(dom_arr)
        dfrange = maxdom - mindom
        diffs   = np.abs(np.diff(dom_arr))
        modindx = (np.sum(diffs) / dfrange) if dfrange > 0 and len(diffs) > 0 else 0.0
    else:
        meandom = mindom = maxdom = dfrange = modindx = 0.0

    return {
        'meanfreq': meanfreq, 'sd': sd, 'median': median,
        'Q25': Q25, 'Q75': Q75, 'IQR': IQR,
        'skew': skew, 'kurt': kurt, 'sp.ent': sp_ent,
        'sfm': sfm_val, 'mode': mode, 'centroid': centroid,
        'meanfun': meanfun, 'minfun': minfun, 'maxfun': maxfun,
        'meandom': meandom, 'mindom': mindom, 'maxdom': maxdom,
        'dfrange': dfrange, 'modindx': modindx,
    }


# ── Prediction ────────────────────────────────────────────────────────────────
from pitch_safety_filter import apply_pitch_safety_filter


def predict_gender(audio_path: str, features: dict) -> dict:
    """Runs the primary Wav2Vec2-XLSR gender model, then applies the pitch
    safety filter. Returns the same response shape the old SVM/GBM/RF
    ensemble used to (svm/gbm/rf/ensemble/decision/features) for API
    backward compatibility — svm/gbm/rf now all mirror the single primary
    model's result."""
    primary = gender_verifier.classify_gender(audio_path)
    primary_label = primary['label']
    primary_conf = primary['confidence'] / 100.0
    logger.info(f"[PRIMARY] Wav2Vec2 verdict: {primary_label} {primary['confidence']}%")

    meanfun_hz = features['meanfun'] * 1000
    meanfreq_hz = features['meanfreq'] * 1000

    final_label, final_conf = apply_pitch_safety_filter(primary_label, primary_conf, meanfun_hz, meanfreq_hz)

    if final_label == 'manual_review':
        logger.info(f"[MANUAL REVIEW] Ambiguous voice. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz, Conf: {final_conf*100:.1f}%.")
    elif final_label != primary_label:
        logger.info(f"[PITCH FILTER] Override applied. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz (Male range).")

    model_output = {'label': final_label, 'confidence': float(final_conf) * 100}

    return {
        'svm': dict(model_output),
        'gbm': dict(model_output),
        'rf': dict(model_output),
        'ensemble': {
            'label':      final_label,
            'confidence': float(final_conf) * 100,
            'male_votes': 3 if final_label == 'male' else 0,
            'total_votes': 3,
        },
        'decision': 'accept' if final_label == 'female' else ('uncertain' if final_label == 'manual_review' else 'reject'),
        'features': {
            'meanfun_hz':  round(meanfun_hz, 1),
            'meanfreq_hz': round(meanfreq_hz, 1),
            'IQR':         round(features['IQR'], 4),
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    advisor_name: str = Form("Unknown Advisor")
):
    """
    Upload audio → save to disk → extract features → predict gender → notify Telegram.
    Heavy processing runs in a worker thread (via run_in_threadpool) so concurrent
    requests are actually processed in parallel instead of serializing on the
    single asyncio event loop — GLOBAL_PROCESS_LOCK below then caps how many of
    those threads run heavy model inference at once.
    """
    if not gender_verifier.is_loaded():
        raise HTTPException(status_code=503, detail="Primary gender model failed to load. Check server logs.")

    content = file.file.read()
    filename = file.filename
    return await run_in_threadpool(_predict_sync, content, filename, advisor_name)


def _predict_sync(content: bytes, filename: str, advisor_name: str) -> JSONResponse:
    file_size_kb = len(content) / 1024

    request_id = uuid.uuid4().hex

    t_start = time.time()
    t_stt, t_df, t_gen = 0.0, 0.0, 0.0
    def log_req(res, err=None):
        try:
            import json
            tot_ms = (time.time() - t_start) * 1000
            try:
                data = json.loads(res.body.decode('utf-8'))
            except:
                data = {}
            dec = data.get('decision', data.get('status', 'unknown'))
            conf = data.get('ensemble', {}).get('confidence', 0.0)
            logger.info(f"ReqID: {request_id} | File: {filename} | Size: {file_size_kb:.1f}KB | Total: {tot_ms:.1f}ms | STT: {t_stt:.1f}ms | Deepfake: {t_df:.1f}ms | Gender: {t_gen:.1f}ms | Decision: {dec} | Conf: {conf} | Err: {err}")
        except Exception:
            pass
        return res

    # ── 2. Determine file extension ───────────────────────────────────────────
    allowed = {'.wav', '.mp3', '.ogg', '.m4a', '.webm', '.flac'}
    original_name = filename or 'recording.wav'
    ext = os.path.splitext(original_name.lower())[1]
    if ext not in allowed:
        ext = '.wav'  # Default to wav (browser sends our encoded WAV)

    # ── 3. Save permanently to recordings/ folder ─────────────────────────────
    # request_id is included (not just the second-precision timestamp) so concurrent
    # requests arriving in the same second never collide on the same filename.
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_name  = f"voice_{timestamp}_{request_id[:8]}{ext}"
    saved_path  = os.path.join(RECORDINGS_DIR, saved_name)

    with open(saved_path, 'wb') as f:
        f.write(content)

    # --- AUDIO NORMALIZATION (Crucial for corrupted/re-encoded files) ---
    try:
        y, sr = safe_load_audio(saved_path, sr=16000)
        
        # ── IMMEDIATE VOLUME CHECK ──
        max_amp = np.max(np.abs(y))
        if max_amp < 0.08:
            logger.info(f"[REJECT] Audio volume too low (Max Amp: {max_amp:.3f})")
            saved_basename = os.path.basename(saved_path)
            _delete_audio(saved_path)
            return log_req(JSONResponse(content={
                'accepted': False,
                'is_female': False,
                'is_ai': False,
                'status': 'rejected_fake',
                'reason': "Audio volume is very low or completely silent. Please speak loudly and clearly.",
                'saved_as': saved_basename,
            }))
            
        norm_name = f"voice_{timestamp}_{request_id[:8]}_norm.wav"
        norm_path = os.path.join(RECORDINGS_DIR, norm_name)
        sf.write(norm_path, y, 16000)
        
        # Replace saved_path with the normalized file
        os.remove(saved_path)
        saved_path = norm_path
        saved_name = norm_name
    except Exception as e:
        logger.exception(f"[WARN] Failed to normalize audio: {e}")

    logger.info(f"[SAVE] Recording saved: {saved_path} ({file_size_kb:.1f} KB)")

    # --- STT HUMAN AUDIBILITY CHECK ---
    try:
        with GLOBAL_PROCESS_LOCK:
            t_stt_start = time.time()
            stt_audio, _ = safe_load_audio(saved_path, sr=16000)
            inputs = stt_processor(stt_audio, sampling_rate=16000, return_tensors='pt').to(STT_DEVICE)
            with torch.no_grad():
                # no_repeat_ngram_size/repetition_penalty guard against Whisper's known repetition-loop
                # degeneration on ambiguous audio (it can otherwise repeat a phrase for hundreds of
                # tokens, ballooning latency from ~2s to 20s+ and producing garbage transcriptions).
                predicted_ids = stt_model.generate(
                    inputs.input_features,
                    max_new_tokens=200,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3,
                )

            transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        words = transcription.split()
        meaningful_words = [w for w in words if len(w) >= 3]
        t_stt = (time.time() - t_stt_start) * 1000
        logger.info(f"[STT] Transcription: '{transcription}' (Using Whisper)")
        
        if len(meaningful_words) < 5:
            logger.info(f"[REJECT] Audio unintelligible. Words: {len(meaningful_words)}")
            saved_basename = os.path.basename(saved_path)
            _delete_audio(saved_path)
            return log_req(JSONResponse(content={
                'accepted': False,
                'is_female': False,
                'is_ai': False,
                'status': 'rejected_fake',
                'reason': f"Voice is not clearly audible (Words: {len(meaningful_words)}). Please speak at least 5 meaningful words in any language.",
                'saved_as': saved_basename,
            }))
    except Exception as e:
        logger.exception(f"[WARN] Failed STT check for {saved_path}: {e}")

    # ── 4. Extract features + predict (use saved file directly) ──────────────
    try:
        with GLOBAL_PROCESS_LOCK:
            # Always run feature extraction first
            features = extract_features(saved_path)

            # Check if AI or Human using the advanced ML Model
            t_df_start = time.time()
            ai_result = advanced_deepfake_detector.predict(saved_path)
            t_df = (time.time() - t_df_start) * 1000

            t_gen_start = time.time()
            result   = predict_gender(saved_path, features)
            t_gen = (time.time() - t_gen_start) * 1000

        result['ai'] = ai_result
        result['ai_voice'] = ai_result.get('is_ai', False)
        if ai_result.get('status') in ['model_error', 'processing_error']:
            result['ai_error'] = ai_result.get('reason', 'Failed to load deepfake model')
        result['email_configured'] = config.email_configured()

        if ai_result.get('is_ai'):
            reason_str = ai_result.get('reason', f"AI/Synthetic voice detected ({ai_result.get('confidence')}%)")
            # The deepfake model still has a residual false-positive rate on real voices
            # outside its training distribution, so a positive here goes to manual_review
            # instead of an outright reject — a human makes the final call instead of an
            # unverified model auto-rejecting a real user.
            logger.info(f"[MANUAL REVIEW] Deepfake model flagged audio, escalating instead of auto-rejecting. Reason: {reason_str}")
            kept_path = _keep_for_manual_review(saved_path)

            result['accepted'] = False
            result['status'] = 'manual_review'
            result['decision'] = 'uncertain'
            result['reason'] = reason_str
            result['saved_as'] = os.path.basename(kept_path)
            result['saved_kb'] = round(file_size_kb, 1)
            _dispatch_email_notification(result, os.path.basename(kept_path), file_size_kb, kept_path)
        else:
            result['saved_kb'] = round(file_size_kb, 1)

            if result['ensemble']['label'] == 'manual_review':
                # Pitch safety filter escalated an ambiguous verdict — keep for human review.
                kept_path = _keep_for_manual_review(saved_path)
                result['saved_as'] = os.path.basename(kept_path)
                _dispatch_email_notification(result, os.path.basename(kept_path), file_size_kb, kept_path)
            else:
                # Auto-decided (clean accept/reject) — no audio is retained.
                result['saved_as'] = saved_name
                _dispatch_email_notification(result, saved_name, file_size_kb, saved_path)
                _delete_audio(saved_path)

        return log_req(JSONResponse(content=result))

    except ValueError as e:
        # Auto-rejected (bad audio) — no audio is retained.
        saved_basename = os.path.basename(saved_path)
        _delete_audio(saved_path)
        logger.exception(f"[REJECT] Audio validation failed: {e}")
        return log_req(JSONResponse(content={
            'accepted': False,
            'is_female': False,
            'reason': str(e),
            'saved_as': saved_basename
        }), err=str(e))
    except Exception as e:
        # Auto-rejected (processing error) — no audio is retained.
        saved_basename = os.path.basename(saved_path)
        _delete_audio(saved_path)
        logger.exception(f"[REJECT] Audio processing error for uploaded file: {e}")
        return log_req(JSONResponse(content={
            'accepted': False,
            'is_female': False,
            'reason': f'Audio processing error: {str(e)}',
            'saved_as': saved_basename
        }), err=str(e))


@app.get("/health")
async def health():
    import psutil
    import time
    
    process = psutil.Process()
    uptime_seconds = time.time() - process.create_time()
    
    hours, remainder = divmod(int(uptime_seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    cpu_usage = psutil.cpu_percent(interval=None)
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    
    try:
        # No audio is retained except pending manual_review cases (everything else
        # is deleted right after processing), so that's what's meaningful to report here.
        rec_count = len([f for f in os.listdir(MANUAL_REVIEW_DIR) if not f.endswith('.npy')])
    except Exception:
        rec_count = 0

    return {
        "status": "ok",
        "api_status": "online",
        "version": app.version,
        "uptime": uptime_str,
        "uptime_seconds": round(uptime_seconds, 1),
        "cpu_usage_percent": cpu_usage,
        "ram_usage_mb": round(ram_usage_mb, 1),
        "models_loaded": gender_verifier.is_loaded(),
        "detailed_model_status": {
            "primary_gender_model": gender_verifier.is_loaded(),
            "stt_whisper": stt_model is not None,
        },
        "email_configured": config.email_configured(),
        "recordings_saved": rec_count,
        "recordings_dir": MANUAL_REVIEW_DIR,
    }


@app.get("/recordings", dependencies=[Depends(require_api_key)])
async def list_recordings():
    """Admin endpoint: lists pending manual_review audio (the only audio this app
    retains — everything else is deleted right after processing)."""
    files = []
    for fname in sorted(os.listdir(MANUAL_REVIEW_DIR), reverse=True):
        if fname.endswith(".npy"):
            continue
        fpath = os.path.join(MANUAL_REVIEW_DIR, fname)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            files.append({
                'filename': fname,
                'size_kb': round(stat.st_size / 1024, 1),
                'saved_at': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            })
    return JSONResponse(content={'total': len(files), 'recordings': files})


from typing import Union

class PredictUrlRequest(BaseModel):
    url: str
    userId: Union[str, int] = "unknown"
    fullname: str = "Unknown"

@app.post("/predict-url", dependencies=[Depends(require_api_key)])
async def predict_from_url(body: PredictUrlRequest):
    """
    Automation endpoint for n8n / AWS Lambda / any webhook.
    Accepts audio URL (S3 pre-signed URL or any HTTP URL), downloads it,
    analyzes gender, saves recording, sends Telegram notification.

    Request body (JSON):
        {
            "url":          "https://s3.amazonaws.com/.../voice.wav",
            "advisor_id":   "12345",
            "advisor_name": "Priya Sharma"   (optional)
        }

    Response:
        { "ensemble": {...}, "svm": {...}, "gbm": {...}, "rf": {...},
          "features": {...}, "advisor_id": "...", "advisor_name": "...",
          "saved_as": "advisor_12345_20260604_153000.wav",
          "is_female": true, "confidence": 91.2 }
    """
    if not gender_verifier.is_loaded():
        raise HTTPException(status_code=503, detail="Primary gender model failed to load. Check server logs.")

    audio_url    = body.url.strip()
    advisor_id   = str(body.userId)
    advisor_name = str(body.fullname)

    if not audio_url:
        raise HTTPException(status_code=400, detail="Missing 'url' field in request body.")

    # Check cache to prevent n8n infinite loops on the same recording
    if audio_url in processed_cache:
        logger.info(f"[CACHE] Returning cached result for Advisor ID: {advisor_id}")
        return JSONResponse(content=processed_cache[audio_url])

    # ── 1. Validate URL to prevent SSRF (no internal/private network fetches) ──
    try:
        _assert_public_url(audio_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # ── 2. Download audio from URL (TLS verification enabled) ─────────────────
    try:
        ctx = ssl.create_default_context()
        req_obj = urllib.request.Request(
            audio_url,
            headers={"User-Agent": "VoiceGenderBot/2.0"}
        )
        with urllib.request.urlopen(req_obj, timeout=30, context=ctx) as resp:
            content = resp.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(e)}")

    file_size_kb = len(content) / 1024

    # Heavy processing runs in a worker thread so concurrent /predict-url calls are
    # actually processed in parallel instead of serializing on the event loop.
    return await run_in_threadpool(_predict_url_sync, content, audio_url, advisor_id, advisor_name, file_size_kb)


def _predict_url_sync(content: bytes, audio_url: str, advisor_id: str, advisor_name: str, file_size_kb: float) -> JSONResponse:
    request_id = uuid.uuid4().hex

    t_start = time.time()
    t_stt, t_df, t_gen = 0.0, 0.0, 0.0
    def log_req(res, err=None):
        try:
            import json
            tot_ms = (time.time() - t_start) * 1000
            try:
                data = json.loads(res.body.decode('utf-8'))
            except:
                data = {}
            dec = data.get('decision', data.get('status', 'unknown'))
            conf = data.get('ensemble', {}).get('confidence', 0.0)
            logger.info(f"ReqID: {request_id} | URL: {audio_url} | Size: {file_size_kb:.1f}KB | Total: {tot_ms:.1f}ms | STT: {t_stt:.1f}ms | Deepfake: {t_df:.1f}ms | Gender: {t_gen:.1f}ms | Decision: {dec} | Conf: {conf} | Err: {err}")
        except Exception:
            pass
        return res

    if file_size_kb < 4.0:
        logger.info(f"[REJECT] Audio file is too small ({file_size_kb:.1f} KB) for Advisor ID: {advisor_id}.")
        res = {
            'decision': 'reject',
            'status': 6,
            'accepted': False,
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'source_url': audio_url,
            'is_female': False,
            'reason': f"Audio file is too small ({file_size_kb:.1f} KB). Minimum 4KB required.",
        }
        _add_to_cache(audio_url, res)
        return log_req(JSONResponse(content=res))

    # ── 2. Determine extension ────────────────────────────────────────────────
    clean_url = audio_url.split("?")[0].lower()   # Remove query params (S3 signed URLs)
    ext = os.path.splitext(clean_url)[1]
    if ext not in {'.wav', '.mp3', '.ogg', '.m4a', '.flac', '.webm'}:
        ext = '.wav'

    # ── 3. Use temp file — no permanent local storage (original is on FriendshipHub) ──
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        logger.info(f"[URL] Downloaded {file_size_kb:.1f} KB -> temp file (no local save)")

        # --- AUDIO NORMALIZATION (Crucial for corrupted/re-encoded files) ---
        try:
            y, sr = safe_load_audio(tmp_path, sr=16000)
            
            # ── IMMEDIATE VOLUME CHECK ──
            max_amp = np.max(np.abs(y))
            if max_amp < 0.20:
                logger.info(f"[REJECT] Audio volume too low (Max Amp: {max_amp:.3f}) for Advisor ID: {advisor_id}")
                res = {
                    'decision': 'reject',
                    'status': 6,
                    'accepted': False,
                    'advisor_id': advisor_id,
                    'advisor_name': advisor_name,
                    'source_url': audio_url,
                    'is_female': False,
                    'reason': "Audio volume is very low or completely silent. Please speak loudly and clearly.",
                }
                _add_to_cache(audio_url, res)
                return log_req(JSONResponse(content=res))
                
            duration = len(y) / sr
            if duration < 4.0:
                logger.info(f"[REJECT] Audio too short ({duration:.1f}s) for Advisor ID: {advisor_id}")
                res = {
                    'decision': 'reject',
                    'status': 6,
                    'accepted': False,
                    'advisor_id': advisor_id,
                    'advisor_name': advisor_name,
                    'source_url': audio_url,
                    'is_female': False,
                    'reason': f"Audio too short ({duration:.1f}s). Please speak clearly for at least 4 seconds.",
                }
                _add_to_cache(audio_url, res)
                return log_req(JSONResponse(content=res))

            norm_path = tmp_path + "_norm.wav"
            sf.write(norm_path, y, 16000)
            os.remove(tmp_path)
            tmp_path = norm_path
        except Exception as e:
            logger.exception(f"[WARN] Failed to normalize audio: {e}")

        # --- STT ONE-WORD REJECT CHECK ---
        try:
            with GLOBAL_PROCESS_LOCK:
                t_stt_start = time.time()
                stt_audio, _ = safe_load_audio(tmp_path, sr=16000)
                inputs = stt_processor(stt_audio, sampling_rate=16000, return_tensors='pt').to(STT_DEVICE)
                with torch.no_grad():
                    predicted_ids = stt_model.generate(
                        inputs.input_features,
                        max_new_tokens=200,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.3,
                    )

                transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            words = transcription.split()
            meaningful_words = [w for w in words if len(w) >= 3]
            t_stt = (time.time() - t_stt_start) * 1000
            logger.info(f"[STT] Transcription: '{transcription}' (Using Whisper)")
            
            if len(meaningful_words) < 5:
                logger.info(f"[REJECT] Audio unintelligible. Words: {len(meaningful_words)}")
                res = {
                    'decision': 'reject',
                    'status': 6,
                    'accepted': False,
                    'advisor_id': advisor_id,
                    'advisor_name': advisor_name,
                    'source_url': audio_url,
                    'is_female': False,
                    'reason': f"Voice is not clearly audible (Words: {len(meaningful_words)}). Please speak at least 5 meaningful words in any language.",
                }
                _add_to_cache(audio_url, res)
                return log_req(JSONResponse(content=res))
            else:
                logger.info(f"[STT] Transcribed ({len(words)} words): {transcription}")
        except Exception as e:
            logger.exception(f"[WARN] STT Transcription failed: {e}")

        # ── 3b. Replay-attack check: same audio previously submitted under a
        # different advisor_id? Skip the expensive deepfake + gender model
        # calls entirely if so — the decision is already made.
        try:
            with GLOBAL_PROCESS_LOCK:
                current_fp = fingerprint.compute_fingerprint(tmp_path)
            duplicate = fingerprint_store.find_cross_advisor_match(current_fp, advisor_id)
        except Exception as e:
            logger.exception(f"[WARN] Fingerprint check failed for Advisor ID: {advisor_id}: {e}")
            current_fp = None
            duplicate = None

        if duplicate is not None:
            reason_str = f"Replay Attack Detected: this audio was previously submitted under a different Advisor ID ({duplicate['advisor_id']})."
            logger.info(f"[MANUAL REVIEW] {reason_str} Current Advisor ID: {advisor_id}. Hamming distance: {duplicate.get('hamming_distance')}.")
            kept_path = _keep_for_manual_review(tmp_path)  # tmp_path itself no longer exists after this

            result = {
                'svm':      {'label': 'manual_review', 'confidence': 0.0},
                'gbm':      {'label': 'manual_review', 'confidence': 0.0},
                'rf':       {'label': 'manual_review', 'confidence': 0.0},
                'ensemble': {'label': 'manual_review', 'confidence': 0.0, 'male_votes': 0, 'total_votes': 3},
                'ai':       {'is_ai': False, 'confidence': 0.0, 'reason': reason_str, 'status': 'success'},
                'status': 'manual_review',
                'request_id': request_id,
                'advisor_id': advisor_id,
                'advisor_name': advisor_name,
                'reason': reason_str,
                'source_url': audio_url,
            }
            _dispatch_email_notification(result, f"{advisor_name} (ID:{advisor_id})", file_size_kb, kept_path)

            n8n_result = {
                'decision': 'uncertain',
                'status': 1,
                'accepted': False,
                'advisor_id': advisor_id,
                'advisor_name': advisor_name,
                'source_url': audio_url,
                'is_female': False,
                'reason': 'This audio matches a previously submitted recording under a different advisor. Sent for manual review.'
            }

            _add_to_cache(audio_url, n8n_result)
            return log_req(JSONResponse(content=n8n_result))

        # ── 4. Extract features + predict ─────────────────────────────────────
        with GLOBAL_PROCESS_LOCK:
            t_df_start = time.time()
            ai_result = advanced_deepfake_detector.predict(tmp_path)
            t_df = (time.time() - t_df_start) * 1000
            
            features = extract_features(tmp_path)

            t_gen_start = time.time()
            result   = predict_gender(tmp_path, features)
            t_gen = (time.time() - t_gen_start) * 1000
        result['ai'] = ai_result
        result['ai_voice'] = ai_result.get('is_ai', False)

        if ai_result.get('is_ai'):
            reason_str = ai_result.get('reason', f"AI/Synthetic voice detected ({ai_result.get('confidence')}%)")
            # Same reasoning as /predict: the deepfake model still has a residual false-positive
            # rate on real voices outside its training distribution, so escalate to manual_review
            # instead of auto-rejecting a real advisor.
            logger.info(f"[MANUAL REVIEW] Deepfake model flagged audio for Advisor ID: {advisor_id}, escalating instead of auto-rejecting. Reason: {reason_str}")
            kept_path = _keep_for_manual_review(tmp_path)  # tmp_path itself no longer exists after this

            result['status'] = 'manual_review'
            result['request_id'] = request_id
            result['advisor_id'] = advisor_id
            result['advisor_name'] = advisor_name
            result['reason'] = reason_str
            result['source_url'] = audio_url
            _dispatch_email_notification(result, f"{advisor_name} (ID:{advisor_id})", file_size_kb, kept_path)

            n8n_result = {
                'decision': 'uncertain',
                'status': 1,
                'accepted': False,
                'advisor_id': advisor_id,
                'advisor_name': advisor_name,
                'source_url': audio_url,
                'is_female': False,
                'reason': reason_str
            }

            if current_fp is not None:
                try:
                    fingerprint_store.store_fingerprint(current_fp, advisor_id, advisor_name, result.get('status', 'manual_review'))
                except Exception as e:
                    logger.exception(f"[WARN] Failed to store fingerprint for Advisor ID: {advisor_id}: {e}")
            _add_to_cache(audio_url, n8n_result)
            return log_req(JSONResponse(content=n8n_result))

        label = result['ensemble']['label']
        display_name = f"{advisor_name} (ID:{advisor_id})"

        # ── 4b. Name Gender Detection ─────────────────────────────────────────
        first_name = advisor_name.split(" ")[0].capitalize()
        name_gender = gender_detector.get_gender(first_name)
        
        gender_mismatch = False
        if label == 'female' and name_gender in ['male', 'mostly_male']:
            gender_mismatch = True
        elif label == 'male' and name_gender in ['female', 'mostly_female']:
            gender_mismatch = True

        # ── 5. REJECT male voice — no Email, no further action ─────────────
        if label == 'male':
            logger.info(f"[REJECT] Male voice detected for Advisor ID: {advisor_id} - rejected, no Email sent.")
            n8n_result = {
                'decision':     'reject',
                'status':       6,
                'accepted':     False,
                'advisor_id':   advisor_id,
                'advisor_name': advisor_name,
                'source_url':   audio_url,
                'is_female':    False,
                'reason':       'Male voice detected but name is female. Rejected for fake identity.' if gender_mismatch else 'Male voice detected. Only female voices are accepted.'
            }
            if current_fp is not None:
                try:
                    fingerprint_store.store_fingerprint(current_fp, advisor_id, advisor_name, 'reject')
                except Exception as e:
                    logger.exception(f"[WARN] Failed to store fingerprint for Advisor ID: {advisor_id}: {e}")
            _add_to_cache(audio_url, n8n_result)
            return log_req(JSONResponse(content=n8n_result))

        # ── 6. Female or Manual Review — enrich result + send Telegram ───────────────────
        result['status']               = 'manual_review' if gender_mismatch else label
        result['decision']             = 'uncertain' if result['status'] == 'manual_review' else ('accept' if result['status'] == 'female' else 'reject')
        result['accepted']             = (result['status'] == 'female')
        result['request_id']           = request_id
        result['advisor_id']           = advisor_id
        result['advisor_name']         = advisor_name
        result['name_gender']          = name_gender
        result['gender_mismatch']      = gender_mismatch
        result['source_url']           = audio_url      # original FriendshipHub URL
        result['saved_kb']             = round(file_size_kb, 1)
        result['is_female']            = True
        result['email_configured']     = config.email_configured()
        result['ai_voice']             = ai_result.get('is_ai', False)

        # ── 7. Email notification (background) + audio retention ───────────
        if result['status'] == 'manual_review':
            kept_path = _keep_for_manual_review(tmp_path)  # tmp_path itself no longer exists after this
            _dispatch_email_notification(result, display_name, file_size_kb, kept_path)
        else:
            # Auto-decided (clean accept) — no audio is retained; finally: below deletes tmp_path.
            _dispatch_email_notification(result, display_name, file_size_kb, tmp_path)

        n8n_result = {
            'decision': result.get('decision', 'reject'),
            'status': 3 if result.get('decision') == 'accept' else (1 if result.get('decision') == 'uncertain' else 6),
            'accepted': result.get('accepted', False),
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'source_url': audio_url,
            'is_female': result.get('is_female', False),
            'reason': 'Voice processed successfully.'
        }

        if current_fp is not None:
            try:
                fingerprint_store.store_fingerprint(current_fp, advisor_id, advisor_name, result.get('decision', 'unknown'))
            except Exception as e:
                logger.exception(f"[WARN] Failed to store fingerprint for Advisor ID: {advisor_id}: {e}")
        _add_to_cache(audio_url, n8n_result)

        return log_req(JSONResponse(content=n8n_result))

    except ValueError as e:
        logger.exception(f"[REJECT] Audio validation failed for Advisor ID: {advisor_id}: {e}")
        return log_req(JSONResponse(content={
            'decision': 'reject',
            'status': 6,
            'accepted': False,
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'source_url': audio_url,
            'is_female': False,
            'reason': str(e),
        }), err=str(e))
    except Exception as e:
        logger.exception(f"[REJECT] Audio processing error for Advisor ID: {advisor_id}: {e}")
        return log_req(JSONResponse(content={
            'decision': 'reject',
            'status': 6,
            'accepted': False,
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'source_url': audio_url,
            'is_female': False,
            'reason': f'Audio processing error: {str(e)}',
        }), err=str(e))

    finally:
        # Always clean up the temp file and its embedding cache (no-op if it was
        # already moved into MANUAL_REVIEW_DIR above).
        if tmp_path and os.path.exists(tmp_path):
            _delete_audio(tmp_path)
            logger.info(f"[URL] Temp file cleaned up: {tmp_path}")


@app.get("/api/admin/metrics", dependencies=[Depends(require_api_key)])
async def admin_metrics():
    import os, re
    log_path = os.path.join(config.LOGS_DIR, "application.log")
    total = 0
    accepted = 0
    rejected = 0
    total_ms = 0.0
    
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "[INFO] - ReqID:" in line:
                        total += 1
                        if "Decision: accept" in line:
                            accepted += 1
                        elif "Decision: reject" in line:
                            rejected += 1
                        
                        match = re.search(r"Total: (\d+\.\d+)ms", line)
                        if match:
                            total_ms += float(match.group(1))
        except Exception:
            pass
            
    avg_latency = (total_ms / total) if total > 0 else 0.0
    
    return {
        "total_requests": total,
        "accepted": accepted,
        "rejected": rejected,
        "avg_latency_ms": round(avg_latency, 1)
    }

@app.get("/api/admin/recent-events", dependencies=[Depends(require_api_key)])
async def admin_recent_events():
    import os, re
    log_path = os.path.join(config.LOGS_DIR, "application.log")
    events = []
    
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in reversed(lines):
                    if "[INFO] - ReqID:" in line:
                        ts_match = re.search(r"^([\d\-]+\s[\d:]+)", line)
                        ts = ts_match.group(1) if ts_match else "Unknown"
                        
                        req_match = re.search(r"ReqID: ([a-zA-Z0-9\-]+)", line)
                        req_id = req_match.group(1) if req_match else "Unknown"
                        
                        dec_match = re.search(r"Decision: ([a-zA-Z]+)", line)
                        dec = dec_match.group(1) if dec_match else "Unknown"
                        
                        lat_match = re.search(r"Total: ([\d\.]+)ms", line)
                        lat = f"{lat_match.group(1)}ms" if lat_match else "Unknown"
                        
                        events.append({
                            "timestamp": ts,
                            "request_id": req_id,
                            "decision": dec,
                            "latency": lat,
                            "status": "Accepted" if dec == "accept" else "Rejected"
                        })
                        if len(events) >= 50:
                            break
        except Exception:
            pass
            
    return {"events": events}

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003, reload=False)
