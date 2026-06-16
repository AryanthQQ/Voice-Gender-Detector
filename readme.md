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