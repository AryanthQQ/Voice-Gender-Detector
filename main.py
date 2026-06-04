"""
Voice Gender Detection - FastAPI Backend
- Accepts audio files (WAV/MP3/OGG etc.)
- Extracts 20 acoustic features via librosa/soundfile
- Runs SVM + GBM + Random Forest ensemble prediction
- Auto-saves every recording to recordings/ folder
- Sends Telegram notification to admin with verification result
"""
import os
import shutil
import tempfile
import threading
import urllib.request
import urllib.parse
import json
from datetime import datetime

import numpy as np
import joblib
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

import config

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Gender Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ───────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    svm_model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
    gbm_model = joblib.load(os.path.join(MODELS_DIR, 'gbm_model.pkl'))
    rf_model  = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
    scaler    = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    FEATURES  = joblib.load(os.path.join(MODELS_DIR, 'features.pkl'))
    print("[OK] All models loaded successfully!")
except Exception as e:
    print(f"[ERR] Error loading models: {e}")
    svm_model = gbm_model = rf_model = scaler = None

# ── Recordings directory ──────────────────────────────────────────────────────
RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), config.RECORDINGS_DIR)
os.makedirs(RECORDINGS_DIR, exist_ok=True)
print(f"[OK] Recordings will be saved to: {RECORDINGS_DIR}")

# ── Telegram Notifier ─────────────────────────────────────────────────────────
class TelegramNotifier:
    """Send Telegram messages using Bot API — stdlib only, no extra packages."""

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.base    = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(self, text: str) -> bool:
        """Send a message. Returns True on success."""
        try:
            payload = urllib.parse.urlencode({
                'chat_id':    self.chat_id,
                'text':       text,
                'parse_mode': 'HTML',
            }).encode()
            req = urllib.request.Request(self.base, data=payload, method='POST')
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                return result.get('ok', False)
        except Exception as ex:
            print(f"[TELEGRAM] Send failed: {ex}")
            return False


_notifier = None
if config.telegram_configured():
    _notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)
    print("[OK] Telegram notifier ready.")
else:
    print("[WARN] Telegram not configured. Edit .env to add BOT_TOKEN and CHAT_ID.")


def _build_telegram_message(result: dict, filename: str, file_size_kb: float) -> str:
    """Build a rich Telegram HTML notification message."""
    ens    = result['ensemble']
    svm    = result['svm']
    gbm    = result['gbm']
    rf     = result['rf']
    feats  = result['features']
    label  = ens['label']
    conf   = ens['confidence']
    votes  = ens['male_votes']
    now    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if label == 'female':
        verdict_line = "VERDICT: <b>FEMALE VERIFIED</b>"
        verdict_icon = "✅"
        status_emoji = "👩"
    else:
        verdict_line = "VERDICT: <b>MALE DETECTED</b>"
        verdict_icon = "🔵"
        status_emoji = "👨"

    female_votes = 3 - votes
    vote_summary = f"{female_votes}/3 Female" if label == 'female' else f"{votes}/3 Male"

    msg = (
        f"🎙️ <b>Voice Gender Verification Alert</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>Time:</b> {now}\n"
        f"🔊 <b>File:</b> <code>{filename}</code>\n"
        f"📁 <b>Size:</b> {file_size_kb:.1f} KB\n\n"
        f"{verdict_icon} {status_emoji} {verdict_line}\n"
        f"<b>Confidence:</b> {conf:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Model Breakdown:</b>\n"
        f"  • SVM:            {svm['label'].title()} ({svm['confidence']:.0f}%)\n"
        f"  • Gradient Boost: {gbm['label'].title()} ({gbm['confidence']:.0f}%)\n"
        f"  • Random Forest:  {rf['label'].title()} ({rf['confidence']:.0f}%)\n"
        f"  • Ensemble Vote:  {vote_summary}\n\n"
        f"<b>Voice Analysis:</b>\n"
        f"  • Avg Fundamental Freq: {feats['meanfun_hz']} Hz\n"
        f"  • Mean Frequency:       {feats['meanfreq_hz']} Hz\n"
        f"  • Variability (IQR):    {feats['IQR']}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Auto-verified by Voice Gender AI v2.0</i>"
    )
    return msg


def _notify_async(result: dict, filename: str, file_size_kb: float):
    """Send Telegram notification in background thread (non-blocking)."""
    if _notifier is None:
        return
    label = result['ensemble']['label']
    if config.NOTIFY_ON == 'female' and label != 'female':
        return  # Only notify for female if configured
    msg = _build_telegram_message(result, filename, file_size_kb)
    ok  = _notifier.send(msg)
    print(f"[TELEGRAM] Notification {'sent' if ok else 'FAILED'} for {filename} ({label})")


# ── Feature Extraction ────────────────────────────────────────────────────────
def extract_features(audio_path: str) -> dict:
    """Extract 20 acoustic features matching the training dataset."""
    import soundfile as sf

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
        print(f"[WARN] soundfile load failed ({load_err}), trying librosa fallback...")
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

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
    mode_idx = np.argmax(power)
    mode = freqs_filt[mode_idx] / 1000.0
    centroid = np.sum(freqs_filt * power_norm) / 1000.0

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
    except Exception:
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
        modindx = (np.sum(diffs) / (dfrange + 1e-10)) if len(diffs) > 0 else 0.0
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
def predict_gender(features: dict) -> dict:
    """Run all 3 models and return ensemble prediction."""
    feat_vec    = np.array([[features[f] for f in FEATURES]])
    feat_scaled = scaler.transform(feat_vec)

    svm_prob = svm_model.predict_proba(feat_vec)[0]
    svm_pred = int(np.argmax(svm_prob))

    gbm_prob = gbm_model.predict_proba(feat_scaled)[0]
    gbm_pred = int(np.argmax(gbm_prob))

    rf_prob  = rf_model.predict_proba(feat_scaled)[0]
    rf_pred  = int(np.argmax(rf_prob))

    votes      = [svm_pred, gbm_pred, rf_pred]
    male_votes = sum(votes)
    final_label = 'male' if male_votes >= 2 else 'female'

    male_avg_conf   = (svm_prob[1] + gbm_prob[1] + rf_prob[1]) / 3.0
    female_avg_conf = 1.0 - male_avg_conf
    final_conf = male_avg_conf if final_label == 'male' else female_avg_conf

    return {
        'svm': {'label': 'male' if svm_pred == 1 else 'female', 'confidence': float(max(svm_prob)) * 100},
        'gbm': {'label': 'male' if gbm_pred == 1 else 'female', 'confidence': float(max(gbm_prob)) * 100},
        'rf':  {'label': 'male' if rf_pred  == 1 else 'female', 'confidence': float(max(rf_prob))  * 100},
        'ensemble': {
            'label':      final_label,
            'confidence': float(final_conf) * 100,
            'male_votes': male_votes,
            'total_votes': 3,
        },
        'features': {
            'meanfun_hz':  round(features['meanfun'] * 1000, 1),
            'meanfreq_hz': round(features['meanfreq'] * 1000, 1),
            'IQR':         round(features['IQR'], 4),
        },
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload audio → save to disk → extract features → predict gender → notify Telegram.
    """
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first.")

    # ── 1. Read uploaded audio content ────────────────────────────────────────
    content = await file.read()
    file_size_kb = len(content) / 1024

    # ── 2. Determine file extension ───────────────────────────────────────────
    allowed = {'.wav', '.mp3', '.ogg', '.m4a', '.webm', '.flac'}
    original_name = file.filename or 'recording.wav'
    ext = os.path.splitext(original_name.lower())[1]
    if ext not in allowed:
        ext = '.wav'  # Default to wav (browser sends our encoded WAV)

    # ── 3. Save permanently to recordings/ folder ─────────────────────────────
    timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_name  = f"voice_{timestamp}{ext}"
    saved_path  = os.path.join(RECORDINGS_DIR, saved_name)

    with open(saved_path, 'wb') as f:
        f.write(content)
    print(f"[SAVE] Recording saved: {saved_path} ({file_size_kb:.1f} KB)")

    # ── 4. Extract features + predict (use saved file directly) ──────────────
    try:
        features = extract_features(saved_path)
        result   = predict_gender(features)

        # Add saved filename to result for frontend display
        result['saved_as'] = saved_name
        result['saved_kb'] = round(file_size_kb, 1)
        result['telegram_configured'] = config.telegram_configured()

        # ── 6. Send Telegram notification (background thread) ─────────────────
        t = threading.Thread(
            target=_notify_async,
            args=(result, saved_name, file_size_kb),
            daemon=True
        )
        t.start()

        return JSONResponse(content=result)

    except Exception as e:
        # Keep failed recording for debugging but mark it
        err_path = saved_path.replace(ext, f'_FAILED{ext}')
        try:
            os.rename(saved_path, err_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")


@app.get("/health")
async def health():
    rec_count = len([f for f in os.listdir(RECORDINGS_DIR) if not f.endswith('_FAILED.wav')])
    return {
        "status": "ok",
        "models_loaded": svm_model is not None,
        "telegram_configured": config.telegram_configured(),
        "recordings_saved": rec_count,
        "recordings_dir": RECORDINGS_DIR,
    }


@app.get("/recordings")
async def list_recordings():
    """Admin endpoint: list all saved recordings."""
    files = []
    for fname in sorted(os.listdir(RECORDINGS_DIR), reverse=True):
        fpath = os.path.join(RECORDINGS_DIR, fname)
        if os.path.isfile(fpath):
            stat = os.stat(fpath)
            files.append({
                'filename': fname,
                'size_kb': round(stat.st_size / 1024, 1),
                'saved_at': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            })
    return JSONResponse(content={'total': len(files), 'recordings': files})


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
