# Voice Gender Detector 🎙️

**AI-powered voice gender detection using an ensemble of 3 machine learning models.**

Live demo (original R version): [voicegender.herokuapp.com](https://voicegender.herokuapp.com)

---

## Features

- 🎙️ **Record voice** directly in the browser (no upload needed)
- 📁 **Upload WAV / MP3 / OGG / FLAC** audio files
- 🤖 **3 ML models** — SVM (98.9%), Gradient Boosting (98.3%), Random Forest (98.1%)
- 📊 **Ensemble voting** for highest accuracy
- 💾 **Auto-saves** every recording to the `recordings/` folder
- 📲 **Telegram notifications** sent to admin after every verification
- 🔒 Privacy-first: all processing is local, no data sent externally

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.x + FastAPI + Uvicorn |
| ML | scikit-learn (SVM, GBM, Random Forest) |
| Audio | librosa + soundfile |
| Frontend | Vanilla HTML/CSS/JS (dark theme) |
| Notifications | Telegram Bot API |

---

## Quick Start

### 1. Install dependencies

```bash
pip install fastapi uvicorn scikit-learn numpy librosa soundfile python-multipart joblib
```

### 2. Train the ML models

```bash
python train_model.py
```

This reads `voice.csv` and saves 3 trained models + scaler to `models/`.

### 3. Configure Telegram (optional)

Copy the example env file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```
TELEGRAM_BOT_TOKEN=your_bot_token_from_@BotFather
TELEGRAM_CHAT_ID=your_chat_id_from_@userinfobot
NOTIFY_ON=all        # or 'female' to only notify for female detections
```

> **How to get credentials:**
> 1. Open Telegram → search `@BotFather` → `/newbot` → copy the token
> 2. Search `@userinfobot` → start it → copy your Chat ID

### 4. Run the server

```bash
python main.py
```

Open **http://localhost:8000** in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/predict` | Upload audio → returns gender prediction JSON |
| `GET` | `/health` | Server health + recording count |
| `GET` | `/recordings` | List all saved recordings (admin) |

### Example `/predict` response

```json
{
  "ensemble": { "label": "female", "confidence": 91.2, "male_votes": 0, "total_votes": 3 },
  "svm":      { "label": "female", "confidence": 89.3 },
  "gbm":      { "label": "female", "confidence": 94.1 },
  "rf":       { "label": "female", "confidence": 82.5 },
  "features": { "meanfun_hz": 198.0, "meanfreq_hz": 212.0, "IQR": 0.0891 },
  "saved_as": "voice_20260604_113522.wav",
  "saved_kb": 96.0,
  "telegram_configured": true
}
```

---

## Telegram Notification Format

When a voice is analyzed, admin receives:

```
🎙️ Voice Gender Verification Alert
━━━━━━━━━━━━━━━━━━━━━━
📅 Time: 2026-06-04 11:35:22
🔊 File: voice_20260604_113522.wav
📁 Size: 96.0 KB

✅ 👩 VERDICT: FEMALE VERIFIED
Confidence: 91.2%
━━━━━━━━━━━━━━━━━━━━━━
Model Breakdown:
  • SVM:            Female (89%)
  • Gradient Boost: Female (94%)
  • Random Forest:  Female (83%)
  • Ensemble Vote:  3/3 Female

Voice Analysis:
  • Avg Fundamental Freq: 198 Hz
  • Mean Frequency:       212 Hz
  • Variability (IQR):    0.0891
━━━━━━━━━━━━━━━━━━━━━━
Auto-verified by Voice Gender AI v2.0
```

---

## Dataset

The model is trained on 3,168 voice samples (50% male, 50% female) with 20 acoustic features:

`meanfreq`, `sd`, `median`, `Q25`, `Q75`, `IQR`, `skew`, `kurt`, `sp.ent`, `sfm`, `mode`, `centroid`, `meanfun`, `minfun`, `maxfun`, `meandom`, `mindom`, `maxdom`, `dfrange`, `modindx`

Download: [voice.csv](voice.csv)

---

## Model Accuracy

| Model | Train | Test |
|---|---|---|
| SVM (RBF) | ~99% | **98.9%** |
| Gradient Boosting | ~99% | **98.3%** |
| Random Forest | ~99% | **98.1%** |
| Ensemble (majority vote) | — | **~99%** |

---

## Project Structure

```
voice-gender-master/
├── main.py              # FastAPI backend (predict + save + notify)
├── train_model.py       # Train ML models from voice.csv
├── config.py            # Load .env settings
├── .env.example         # Template for Telegram credentials
├── .gitignore           # Excludes .env, recordings/, models/
├── voice.csv            # Training dataset (3,168 samples)
├── models/              # Trained model files (git-ignored, regenerate with train_model.py)
├── recordings/          # Saved audio files (git-ignored, user data)
├── static/
│   └── index.html       # Frontend UI (dark theme, responsive)
└── Web/                 # Original R Shiny app (reference)
```

---

## Original R Version

The original R Shiny app is in the `Web/` directory. It requires R with packages:
`shiny`, `shinyjs`, `RJSONIO`, `RCurl`, `warbleR`, `tuneR`, `seewave`

---

## License & Credits

- Original R project by [Kory Becker](http://primaryobjects.com/kory-becker)
- Python/FastAPI rebuild + Telegram integration added 2026
- Dataset: 3,168 voice samples from Harvard-Haskins, TSP, VoxForge, CMU-ARCTIC
