---
license: mit
datasets:
- fixie-ai/librispeech_asr
language:
- en
base_model:
- facebook/wav2vec2-base
pipeline_tag: audio-classification
metrics:
- accuracy
library_name: transformers
tags:
- voice_phishing
- audio_classification
---
# Voice Detection AI - Real vs AI Audio Classifier

![image/webp](https://cdn-uploads.huggingface.co/production/uploads/674d0f7d7951ab7c4e09f748/-nSLK7WFumAlfv6X69TsW.webp)

### **Model Overview**
This model is a fine-tuned Wav2Vec2-based audio classifier capable of distinguishing between **real human voices** and **AI-generated voices**. It has been trained on a dataset containing samples from various TTS models and real human audio recordings.

---

### **Model Details**
- **Architecture:** Wav2Vec2ForSequenceClassification
- **Fine-tuned on:** Custom dataset with real and AI-generated audio
- **Classes:**
  1. Real Human Voice
  2. AI-generated (e.g., Melgan, DiffWave, etc.)
- **Input Requirements:**
  - Audio format: `.wav`, `.mp3`, etc.
  - Sample rate: 16kHz
  - Max duration: 10 seconds (longer audios are truncated, shorter ones are padded)

---


### **Performance**
- **Robustness:** Successfully classifies across multiple AI-generation models.
- **Limitations:** Struggles with certain unseen AI-generation models (e.g., ElevenLabs).

---

### **How to Use**

#### **1. Install Dependencies**
Make sure you have `transformers` and `torch` installed:
```bash
pip install transformers torch torchaudio
```
##  Usage
### Here's how to use VoiceGUARD for audio classification:
```
import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torchaudio

# Load model and processor
model_name = "Mrkomiljon/voiceGUARD"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2Processor.from_pretrained(model_name)

# Load audio
waveform, sample_rate = torchaudio.load("path_to_audio_file.wav")

# Resample if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)

# Preprocess
inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

# Inference
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)

# Map to label
labels = ["Real Human Voice", "AI-generated"]
prediction = labels[predicted_ids.item()]
print(f"Prediction: {prediction}")
```
## Training Procedure
- Data Collection: Compiled a balanced dataset of real human voices and AI-generated samples from various TTS models.
- Preprocessing: Standardized audio formats, resampled to 16 kHz, and adjusted durations to 10 seconds.
- Fine-Tuning: Utilized the Wav2Vec2 architecture for sequence classification, training for 3 epochs with a learning rate of 1e-5.
## Evaluation
- Metrics: Accuracy, Precision, Recall
- Results: Achieved 99.8% validation accuracy on the test set.
## Limitations and Future Work
- While VoiceGUARD performs robustly across known AI-generation models, it may encounter challenges with novel or unseen models.
- Future work includes expanding the training dataset with samples from emerging TTS technologies to enhance generalization.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
* Special thanks to the developers of the [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) model and the contributors to the datasets used in this project.
* View the complete project on [GitHub](https://github.com/Mrkomiljon/VoiceGUARD2)

---

## Deploying the API on Ubuntu (systemd)

This runs the FastAPI app (`main.py`) as a managed `systemd` service. Target layout:
app + venv under `/opt/voiceguard`, persistent data (recordings, logs, model
caches) under `/data/voiceguard`, running as a dedicated `voiceguard` user.

### 1. System prerequisites

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv git
```

`ffmpeg` is **not** required as a system package — `imageio-ffmpeg` downloads
its own static binary the first time audio needs it. Install `ffmpeg` via
apt yourself only if you'd rather avoid that runtime download.

### 2. Create the service user and directories

```bash
sudo useradd --system --home /opt/voiceguard --shell /usr/sbin/nologin voiceguard
sudo mkdir -p /opt/voiceguard /data/voiceguard
sudo chown -R voiceguard:voiceguard /opt/voiceguard /data/voiceguard
```

### 3. Deploy the code

```bash
sudo -u voiceguard git clone https://github.com/AryanthQQ/Voice-Gender-Detector.git /opt/voiceguard
cd /opt/voiceguard
sudo -u voiceguard python3.11 -m venv venv
sudo -u voiceguard ./venv/bin/pip install -r requirements.txt
```

**On a GPU box:** `pip install torch` from `requirements.txt` can silently give
you a CPU-only wheel. Install the CUDA build for your driver *first* (check
with `nvidia-smi`), then install the rest:

```bash
sudo -u voiceguard ./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
sudo -u voiceguard ./venv/bin/pip install -r requirements.txt
sudo -u voiceguard ./venv/bin/python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

The Whisper STT model, the deepfake detector, and the secondary gender
verifier all auto-select CUDA when `torch.cuda.is_available()` — no config
needed beyond having a working GPU torch install. Once confirmed, raise
`MAX_CONCURRENT_JOBS` in `.env` (start at 8-16 and load-test) — see the
comment in `.env.example`.

### 4. Configure

```bash
sudo -u voiceguard cp .env.example .env
sudo -u voiceguard nano .env
```

Fill in at least:
- `API_KEY` — generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- `STORAGE_BASE=/data/voiceguard`
- `PUBLIC_BASE_URL` — your real domain (e.g. `https://voiceguard.yourapp.com`), used in manual-review email alert links
- `SMTP_*` / `EMAIL_TO` — only if you want manual-review email notifications

### 5. Install and start the systemd service

```bash
sudo cp deploy/voiceguard.service /etc/systemd/system/voiceguard.service
sudo systemctl daemon-reload
sudo systemctl enable --now voiceguard
```

### 6. Verify

```bash
sudo systemctl status voiceguard
journalctl -u voiceguard -f   # first start downloads the Whisper/Wav2Vec2 models — can take a few minutes
curl http://127.0.0.1:8003/health
```

### 7. Expose it

The app binds plain HTTP on `0.0.0.0:8003` with no TLS. Put nginx or caddy in
front for HTTPS on your real domain — that's also what `PUBLIC_BASE_URL`
should point at. Only open port 8003 directly in the firewall if you're not
reverse-proxying:

```bash
sudo ufw allow 8003/tcp
```

### Updating

```bash
cd /opt/voiceguard
sudo -u voiceguard git pull
sudo -u voiceguard ./venv/bin/pip install -r requirements.txt
sudo systemctl restart voiceguard
```