import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path

class AdvancedDeepfakeDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DEEPFAKE] Using device: {self.device}")
        
        # Wav2Vec2 (deep embeddings ke liye)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        self.wav2vec_model.eval()
        
        self.scaler = None
        self.classifier = None
        self.is_trained = False

    def extract_wav2vec_embedding(self, audio_path: str) -> np.ndarray:
        """Wav2Vec2 se deep embedding nikaalta hai"""
        cache_path = audio_path + ".npy"
        if os.path.exists(cache_path):
            return np.load(cache_path)
            
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 10 second chunks mein process karo (memory safe)
            if len(audio) > 16000 * 10:
                audio = audio[:16000 * 10]
            
            inputs = self.feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.wav2vec_model(**inputs)
                # Last hidden state ka mean lete hain (global embedding)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            
            np.save(cache_path, embedding)
            return embedding
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None

    def train_on_your_data(self, real_dir="data/real", fake_dir="data/fake"):
        """Tumhare collected data pe train karega"""
        print("🚀 Starting training with Wav2Vec2 embeddings...")
        
        X = []
        y = []
        
        # Real voices
        print("Processing Real Voices...")
        for file in Path(real_dir).glob("*.*"):
            if file.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
                emb = self.extract_wav2vec_embedding(str(file))
                if emb is not None:
                    X.append(emb)
                    y.append(0)  # 0 = Real
        
        # Fake voices
        print("Processing Fake Voices...")
        for file in Path(fake_dir).glob("*.*"):
            if file.suffix.lower() in ['.wav', '.mp3', '.m4a']:
                emb = self.extract_wav2vec_embedding(str(file))
                if emb is not None:
                    X.append(emb)
                    y.append(1)  # 1 = AI/Fake
        
        X = np.array(X)
        y = np.array(y)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.classifier = xgb.XGBClassifier(
            n_estimators=300, 
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        # Save model
        joblib.dump(self.classifier, "models/wav2vec_xgb.pkl")
        joblib.dump(self.scaler, "models/wav2vec_scaler.pkl")
        
        print(f"✅ Training Complete! Dataset size: {len(y)} samples")
        print(f"   Real: {sum(y==0)} | Fake: {sum(y==1)}")
        return self

    def predict(self, audio_path: str, threshold=0.65):
        """Final prediction function for your agent"""
        if not self.is_trained:
            if os.path.exists("models/wav2vec_xgb.pkl"):
                self.classifier = joblib.load("models/wav2vec_xgb.pkl")
                self.scaler = joblib.load("models/wav2vec_scaler.pkl")
                self.is_trained = True
            else:
                return {"is_ai": False, "confidence": 0.0, "reason": "Model not trained", "status": "error"}
        
        embedding = self.extract_wav2vec_embedding(audio_path)
        if embedding is None:
            return {"is_ai": False, "confidence": 0.0, "reason": "Failed to extract embedding", "status": "error"}

        embedding_scaled = self.scaler.transform([embedding])
        
        prob = self.classifier.predict_proba(embedding_scaled)[0][1]  # Probability of being FAKE
        is_ai = prob >= threshold
        
        confidence = float(round(prob * 100, 1))
        
        if is_ai:
            reason = f"AI Voice Detected (Confidence: {confidence}%)"
            print(f"[REJECT] FAKE! AI DETECTED -> {confidence}%")
        else:
            reason = f"Real Human Voice (AI Prob: {confidence}%)"
            print(f"[OK] REAL HUMAN -> {confidence}%")
        
        return {
            "is_ai": bool(is_ai),
            "confidence": confidence,
            "probability_ai": confidence,
            "reason": reason,
            "status": "success"
        }


if __name__ == "__main__":
    detector = AdvancedDeepfakeDetector()
    detector.train_on_your_data()
