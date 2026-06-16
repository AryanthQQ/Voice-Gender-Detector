"""
Voice Gender Detection - FastAPI Backend
- Accepts audio files (WAV/MP3/OGG etc.)
- Extracts 20 acoustic features via librosa/soundfile
- Runs SVM + GBM + Random Forest ensemble prediction
- Auto-saves every recording to recordings/ folder
- Sends Telegram notification to admin with verification result
"""
import imageio_ffmpeg
import os
import shutil
import tempfile
import threading
import urllib.request
import urllib.parse
import json
from datetime import datetime

# Ensure ffmpeg is available for audioread
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

import numpy as np
import joblib
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from pydantic import BaseModel
import config
from deepfake_detector_v2 import AdvancedDeepfakeDetector
import gender_guesser.detector as gender

# Limit entire request pipeline to 2 concurrent tasks to prevent VPS RAM exhaustion
import threading
GLOBAL_PROCESS_LOCK = threading.Semaphore(2)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Voice Gender Detection API", version="2.0.0")

# Initialize name gender detector
gender_detector = gender.Detector()

# Initialize Advanced Deepfake Detector
advanced_deepfake_detector = AdvancedDeepfakeDetector()

# Initialize Speech-to-Text Model
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
print("[STT] Loading Speech-to-Text model...")
stt_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
stt_model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
stt_model.eval()
print("[STT] Speech-to-Text model loaded successfully!")

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

# ── Simple In-Memory Cache for n8n Loop Protection ────────────────────────────
processed_cache = {}

# ── Telegram Notifier ─────────────────────────────────────────────────────────
class TelegramNotifier:
    """Send Telegram messages using Bot API — stdlib only, no extra packages."""

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.base    = f"https://api.telegram.org/bot{token}/sendMessage"

    def send(self, text: str) -> bool:
        """Send a message. Returns True on success."""
        import ssl
        try:
            # Create SSL context (bypass cert verification for Windows compat)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            payload = urllib.parse.urlencode({
                'chat_id':    self.chat_id,
                'text':       text,
                'parse_mode': 'HTML',
            }).encode()
            req = urllib.request.Request(self.base, data=payload, method='POST')
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
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


def _build_telegram_message(result: dict, filename: str, file_size_kb: float, source_url: str = None) -> str:
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
    elif label == 'manual_review':
        verdict_line = "VERDICT: <b>MANUAL REVIEW NEEDED</b>"
        verdict_icon = "⚠️"
        status_emoji = "🧐"
    else:
        verdict_line = "VERDICT: <b>MALE DETECTED</b>"
        verdict_icon = "🔵"
        status_emoji = "👨"

    female_votes = 3 - votes
    if label == 'female':
        vote_summary = f"{female_votes}/3 Female"
    elif label == 'male':
        vote_summary = f"{votes}/3 Male"
    else:
        vote_summary = f"{female_votes}/3 Female (Ambiguous)"

    # Audio link line — only shown when source URL is available
    audio_link_line = ""
    if source_url:
        audio_link_line = f"🔗 <b>Recording:</b> <a href=\"{source_url}\">▶️ Listen and Verify</a>\n"

    msg = (
        f"🎙️ <b>Voice Gender Verification Alert</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 <b>Time:</b> {now}\n"
        f"🔊 <b>File:</b> <code>{filename}</code>\n"
        f"📁 <b>Size:</b> {file_size_kb:.1f} KB\n"
        f"{audio_link_line}"
        f"\n"
        f"{verdict_icon} {status_emoji} {verdict_line}\n"
        f"<b>Confidence:</b> {conf:.1f}%\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Model Breakdown:</b>\n"
        f"  • AI Check:       {'🔴 Replay Attack' if 'Replay' in result.get('ai', {}).get('reason', '') else '🔴 AI/Deepfake' if result.get('ai', {}).get('is_ai') else '✅ Real Human'}\n"
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
    if label == 'male':
        return  # Do not send notifications for male voices
    if config.NOTIFY_ON == 'female' and label not in ('female', 'manual_review'):
        return  # Only notify for female or manual review if configured
    # Extract source URL if available (from /predict-url flow)
    source_url = result.get('source_url') or None
    msg = _build_telegram_message(result, filename, file_size_kb, source_url=source_url)
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

    # ── AUDIO FILTERING (IMPROVES ACCURACY) ──────────────────────────────────
    # 1. Raw Silence / Blank Noise Detection (Check BEFORE normalization)
    if np.max(np.abs(y)) < 0.05:
        raise ValueError("Audio volume is very low or completely silent. Please speak loudly and clearly.")
        
    # 2. Silence Trimming (Shuru aur aakhir ka blank noise/shanti hatana)
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # 3. Short Audio Rejection
    if len(y) < 16000 * 1.5:
        raise ValueError("Voice is not clear or mostly background noise. Please record in a quiet place.")

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
    if sfm_val > 0.15:
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
            if voiced_ratio < 0.15:
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

    models = [
        ('male' if svm_pred == 1 else 'female', float(max(svm_prob))),
        ('male' if gbm_pred == 1 else 'female', float(max(gbm_prob))),
        ('male' if rf_pred  == 1 else 'female', float(max(rf_prob)))
    ]
    best_model = max(models, key=lambda x: x[1])
    
    final_label = best_model[0]
    final_conf = best_model[1]

    male_votes = sum([svm_pred, gbm_pred, rf_pred])

    # ── PITCH (FREQUENCY) HARD FILTER & MANUAL REVIEW ───────────────────
    meanfun_hz = features['meanfun'] * 1000
    meanfreq_hz = features['meanfreq'] * 1000
    if final_label == 'female':
        if meanfun_hz < 125.0 or meanfreq_hz < 125.0:
            # Definitely Male range (below 125 Hz is clearly male)
            final_label = 'male'
            final_conf = 0.999
            print(f"[PITCH FILTER] Override applied. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz (Male range).")
        elif meanfun_hz < 155.0 or meanfreq_hz < 145.0 or (final_conf * 100) < 70.0:
            # Ambiguous pitch, frequency or low confidence -> send to manager
            final_label = 'manual_review'
            print(f"[MANUAL REVIEW] Ambiguous voice. Pitch: {meanfun_hz:.1f} Hz, MeanFreq: {meanfreq_hz:.1f} Hz, Conf: {final_conf*100:.1f}%.")


    male_votes = sum([svm_pred, gbm_pred, rf_pred])

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
def predict(file: UploadFile = File(...)):
    """
    Upload audio → save to disk → extract features → predict gender → notify Telegram.
    """
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first.")

    # ── 1. Read uploaded audio content ────────────────────────────────────────
    content = file.file.read()
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
        
    # --- AUDIO NORMALIZATION (Crucial for corrupted/re-encoded files) ---
    try:
        import soundfile as sf
        import librosa
        y, sr = librosa.load(saved_path, sr=16000)
        
        # ── IMMEDIATE VOLUME CHECK ──
        max_amp = np.max(np.abs(y))
        if max_amp < 0.20:
            print(f"[REJECT] Audio volume too low (Max Amp: {max_amp:.3f})")
            return JSONResponse(content={
                'accepted': False,
                'is_female': False,
                'is_ai': False,
                'status': 'rejected_fake',
                'reason': "Audio volume is very low or completely silent. Please speak loudly and clearly.",
                'saved_as': os.path.basename(saved_path),
            })
            
        norm_name = f"voice_{timestamp}_norm.wav"
        norm_path = os.path.join(RECORDINGS_DIR, norm_name)
        sf.write(norm_path, y, 16000)
        
        # Replace saved_path with the normalized file
        os.remove(saved_path)
        saved_path = norm_path
        saved_name = norm_name
    except Exception as e:
        print(f"[WARN] Failed to normalize audio: {e}")

    print(f"[SAVE] Recording saved: {saved_path} ({file_size_kb:.1f} KB)")
    
    # --- STT HUMAN AUDIBILITY CHECK ---
    try:
        stt_audio, _ = librosa.load(saved_path, sr=16000)
        inputs = stt_processor(stt_audio, sampling_rate=16000, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits = stt_model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = stt_processor.batch_decode(predicted_ids)[0]
        words = transcription.split()
        print(f"[STT] Transcription for uploaded file: '{transcription}'")
        if len(words) <= 3:
            print(f"[REJECT] Audio unintelligible, only {len(words)} words detected.")
            return JSONResponse(content={
                'accepted': False,
                'is_female': False,
                'is_ai': False,
                'status': 'rejected_fake',
                'reason': f"Voice is not clearly audible (only {len(words)} words detected). Please speak loud and clear.",
                'saved_as': os.path.basename(saved_path),
            })
    except Exception as e:
        print(f"[WARN] Failed STT check for {saved_path}: {e}")

    # ── 4. Extract features + predict (use saved file directly) ──────────────
    try:
        # Always run feature extraction first
        features = extract_features(saved_path)
        
        # Check if AI or Human using the advanced ML Model
        ai_result = advanced_deepfake_detector.predict(saved_path)
        
        result   = predict_gender(features)
        result['ai'] = ai_result
        result['ai_voice'] = ai_result.get('is_ai', False)
        if ai_result.get('status') in ['model_error', 'processing_error']:
            result['ai_error'] = ai_result.get('reason', 'Failed to load deepfake model')
        result['telegram_configured'] = config.telegram_configured()

        if ai_result.get('is_ai'):
            reason_str = ai_result.get('reason', f"AI/Synthetic voice detected ({ai_result.get('confidence')}%)")
            print(f"[REJECT] Spoof/AI Voice detected. Reason: {reason_str}")
            # Keep failed recording but mark it
            err_path = saved_path.replace(ext, f'_AI_FAKE{ext}')
            try: os.rename(saved_path, err_path)
            except: pass
            
            result['accepted'] = False
            result['status'] = 'rejected_fake'
            result['reason'] = reason_str
            result['saved_as'] = os.path.basename(err_path)
            result['saved_kb'] = round(file_size_kb, 1)
        else:
            # Add saved filename to result for frontend display
            result['saved_as'] = saved_name
            result['saved_kb'] = round(file_size_kb, 1)

            # ── 6. Send Telegram notification (background thread) ─────────────────
            t = threading.Thread(
                target=_notify_async,
                args=(result, saved_name, file_size_kb),
                daemon=True
            )
            t.start()

        return JSONResponse(content=result)

    except ValueError as e:
        # Keep failed recording for debugging but mark it
        err_path = saved_path.replace(ext, f'_FAILED{ext}')
        try:
            os.rename(saved_path, err_path)
        except Exception:
            pass
        print(f"[REJECT] Audio validation failed: {e}")
        return JSONResponse(content={
            'accepted': False,
            'is_female': False,
            'reason': str(e),
            'saved_as': os.path.basename(err_path)
        })
    except Exception as e:
        # Keep failed recording for debugging but mark it
        err_path = saved_path.replace(ext, f'_FAILED{ext}')
        try:
            os.rename(saved_path, err_path)
        except Exception:
            pass
        print(f"[REJECT] Audio processing error for uploaded file: {e}")
        return JSONResponse(content={
            'accepted': False,
            'is_female': False,
            'reason': f'Audio processing error: {str(e)}',
            'saved_as': os.path.basename(err_path)
        })


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


from typing import Union

class PredictUrlRequest(BaseModel):
    url: str
    userId: Union[str, int] = "unknown"
    fullname: str = "Unknown"

@app.post("/predict-url")
def predict_from_url(body: PredictUrlRequest):
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
    if svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_model.py first.")

    audio_url    = body.url.strip()
    advisor_id   = str(body.userId)
    advisor_name = str(body.fullname)

    if not audio_url:
        raise HTTPException(status_code=400, detail="Missing 'url' field in request body.")

    # Check cache to prevent n8n infinite loops on the same recording
    if audio_url in processed_cache:
        print(f"[CACHE] Returning cached result for {advisor_name}")
        return JSONResponse(content=processed_cache[audio_url])

    # ── 1. Download audio from URL ────────────────────────────────────────────
    import ssl
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        req_obj = urllib.request.Request(
            audio_url,
            headers={"User-Agent": "VoiceGenderBot/2.0"}
        )
        with urllib.request.urlopen(req_obj, timeout=30, context=ctx) as resp:
            content = resp.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(e)}")

    file_size_kb = len(content) / 1024

    if file_size_kb < 4.0:
        print(f"[REJECT] Audio file is too small ({file_size_kb:.1f} KB) for {advisor_name}.")
        res = {
            'accepted': False,
            'is_female': False,
            'is_ai': False,
            'status': 'rejected_error',
            'reason': f"Audio file is too small ({file_size_kb:.1f} KB). Minimum 4KB required.",
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'saved_kb': round(file_size_kb, 1),
        }
        processed_cache[audio_url] = res
        if len(processed_cache) > 1000:
            processed_cache.pop(next(iter(processed_cache)))
        return JSONResponse(content=res)

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
        print(f"[URL] Downloaded {file_size_kb:.1f} KB -> temp file (no local save)")

        # --- AUDIO NORMALIZATION (Crucial for corrupted/re-encoded files) ---
        try:
            import soundfile as sf
            import librosa
            y, sr = librosa.load(tmp_path, sr=16000)
            
            # ── IMMEDIATE VOLUME CHECK ──
            max_amp = np.max(np.abs(y))
            if max_amp < 0.20:
                print(f"[REJECT] Audio volume too low (Max Amp: {max_amp:.3f}) for {advisor_name}")
                res = {
                    'accepted': False,
                    'is_female': False,
                    'is_ai': False,
                    'status': 'rejected_fake',
                    'reason': "Audio volume is very low or completely silent. Please speak loudly and clearly.",
                    'advisor_id': advisor_id,
                    'advisor_name': advisor_name,
                    'saved_kb': round(file_size_kb, 1),
                }
                processed_cache[audio_url] = res
                if len(processed_cache) > 1000:
                    processed_cache.pop(next(iter(processed_cache)))
                return JSONResponse(content=res)
                
            duration = len(y) / sr
            if duration < 4.0:
                print(f"[REJECT] Audio too short ({duration:.1f}s) for {advisor_name}")
                res = {
                    'accepted': False,
                    'is_female': False,
                    'is_ai': False,
                    'status': 'rejected_fake',
                    'reason': f"Audio too short ({duration:.1f}s). Please speak clearly for at least 4 seconds.",
                    'advisor_id': advisor_id,
                    'advisor_name': advisor_name,
                    'saved_kb': round(file_size_kb, 1),
                }
                processed_cache[audio_url] = res
                if len(processed_cache) > 1000:
                    processed_cache.pop(next(iter(processed_cache)))
                return JSONResponse(content=res)

            norm_path = tmp_path + "_norm.wav"
            sf.write(norm_path, y, 16000)
            os.remove(tmp_path)
            tmp_path = norm_path
        except Exception as e:
            print(f"[WARN] Failed to normalize audio: {e}")

        # --- STT ONE-WORD REJECT CHECK ---
        try:
            stt_audio, _ = librosa.load(tmp_path, sr=16000)
            inputs = stt_processor(stt_audio, sampling_rate=16000, return_tensors='pt', padding=True)
            with torch.no_grad():
                logits = stt_model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = stt_processor.batch_decode(predicted_ids)[0]
            
            words = transcription.split()
            if len(words) <= 3:
                print(f"[REJECT] Audio only contains <=2 words: '{transcription}'.")
                res = {
                    'accepted': False,
                    'is_female': False,
                    'is_ai': False,
                    'status': 'rejected_fake',
                    'reason': f"Audio only contains {len(words)} word(s) ('{transcription}'). Please speak a full clear sentence.",
                    'advisor_id': advisor_id,
                    'advisor_name': advisor_name,
                    'saved_kb': round(file_size_kb, 1),
                }
                processed_cache[audio_url] = res
                if len(processed_cache) > 1000:
                    processed_cache.pop(next(iter(processed_cache)))
                return JSONResponse(content=res)
            else:
                print(f"[STT] Transcribed ({len(words)} words): {transcription}")
        except Exception as e:
            print(f"[WARN] STT Transcription failed: {e}")

        # ── 4. Extract features + predict ─────────────────────────────────────
        ai_result = advanced_deepfake_detector.predict(tmp_path)
        features = extract_features(tmp_path)
        result   = predict_gender(features)
        result['ai'] = ai_result
        result['ai_voice'] = ai_result.get('is_ai', False)

        if ai_result.get('is_ai'):
            reason_str = ai_result.get('reason', f"AI/Synthetic voice detected ({ai_result.get('confidence')}%)")
            print(f"[REJECT] Spoof/AI Voice detected for {advisor_name}. Reason: {reason_str}")
            
            result['accepted'] = False
            result['status'] = 'rejected_fake'
            result['reason'] = reason_str
            result['advisor_id'] = advisor_id
            result['advisor_name'] = advisor_name
            result['saved_kb'] = round(file_size_kb, 1)
            
            processed_cache[audio_url] = result
            if len(processed_cache) > 1000:
                processed_cache.pop(next(iter(processed_cache)))
            return JSONResponse(content=result)

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

        # ── 5. REJECT male voice — no Telegram, no further action ─────────────
        if label == 'male':
            print(f"[REJECT] Male voice detected for {display_name} — rejected, no Telegram sent.")
            res = {
                'accepted':     False,
                'is_female':    False,
                'is_ai':        False,
                'status':       'rejected_male',
                'reason':       'Male voice detected but name is female. Rejected for fake identity.' if gender_mismatch else 'Male voice detected. Only female voices are accepted.',
                'ensemble':     result['ensemble'],
                'svm':          result['svm'],
                'gbm':          result['gbm'],
                'rf':           result['rf'],
                'advisor_id':   advisor_id,
                'advisor_name': advisor_name,
                'name_gender':  name_gender,
                'gender_mismatch': gender_mismatch,
                'saved_kb':     round(file_size_kb, 1),
            }
            processed_cache[audio_url] = res
            if len(processed_cache) > 1000:
                processed_cache.pop(next(iter(processed_cache)))
            return JSONResponse(content=res)

        # ── 6. Female or Manual Review — enrich result + send Telegram ───────────────────
        result['status']               = 'manual_review' if gender_mismatch else label
        result['accepted']             = (result['status'] == 'female')
        result['advisor_id']           = advisor_id
        result['advisor_name']         = advisor_name
        result['name_gender']          = name_gender
        result['gender_mismatch']      = gender_mismatch
        result['source_url']           = audio_url      # original FriendshipHub URL
        result['saved_kb']             = round(file_size_kb, 1)
        result['is_female']            = True
        result['telegram_configured']  = config.telegram_configured()
        result['ai_voice']             = ai_result.get('is_ai', False)

        # ── 7. Telegram notification (background) ─────────────────────────────
        t = threading.Thread(
            target=_notify_async,
            args=(result, display_name, file_size_kb),
            daemon=True
        )
        t.start()

        processed_cache[audio_url] = result
        if len(processed_cache) > 1000:
            processed_cache.pop(next(iter(processed_cache)))

        return JSONResponse(content=result)

    except ValueError as e:
        print(f"[REJECT] Audio validation failed for {advisor_name} (ID: {advisor_id}): {e}")
        return JSONResponse(content={
            'accepted': False,
            'is_female': False,
            'is_ai': False,
            'status': 'rejected_fake',
            'reason': str(e),
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'saved_kb': round(file_size_kb, 1) if 'file_size_kb' in locals() else 0.0,
        })
    except Exception as e:
        print(f"[REJECT] Audio processing error for {advisor_name} (ID: {advisor_id}): {e}")
        return JSONResponse(content={
            'accepted': False,
            'is_female': False,
            'is_ai': False,
            'status': 'rejected_error',
            'reason': f'Audio processing error: {str(e)}',
            'advisor_id': advisor_id,
            'advisor_name': advisor_name,
            'saved_kb': round(file_size_kb, 1) if 'file_size_kb' in locals() else 0.0,
        })

    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
                print(f"[URL] Temp file cleaned up: {tmp_path}")
            except Exception:
                pass


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
