import torch
import librosa
import numpy as np
import io
import torch.nn.functional as F
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

class VoiceDetector:
    def __init__(self):
        print("⏳ Loading Robust AI Detection Model...")
        # Switching to a more stable model for Hackathons
        self.model_name = "padmalcom/wav2vec2-large-fake-voice-detection-v2"
        
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.eval()
            
            # Print labels to debug (Check logs to see what 0 and 1 mean)
            print(f"✅ Model Labels: {self.model.config.id2label}")
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR: Failed to load AI model. {e}")
            raise e

    def preprocess_audio(self, audio_buffer: io.BytesIO, target_sr=16000):
        """
        Robust preprocessing: Resample, Normalize, and Fix Duration.
        """
        audio_buffer.seek(0)
        y, sr = librosa.load(audio_buffer, sr=target_sr)
        
        # 1. Normalize Volume (Crucial for quiet clips)
        y = librosa.util.normalize(y)
        
        # 2. Fix Duration (Model expects ~3-5 seconds)
        # If too short (< 1.5s), loop it.
        if len(y) < target_sr * 1.5:
            tile_factor = int_(np.ceil((target_sr * 1.5) / len(y)))
            y = np.tile(y, tile_factor)
            
        # 3. Limit Duration (If > 10s, take the middle 5s)
        # Long files confuse the model logic if not chunked.
        max_len = target_sr * 5
        if len(y) > max_len:
            start = (len(y) - max_len) // 2
            y = y[start : start + max_len]
            
        return y

    def analyze(self, audio_buffer: io.BytesIO, language: str):
        """
        Analyzes audio with improved threshold logic.
        """
        try:
            # 1. Preprocess
            audio_input = self.preprocess_audio(audio_buffer)
            
            # 2. Prepare Input
            inputs = self.feature_extractor(
                audio_input, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            # 3. Inference
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # 4. Get Probabilities
            probs = F.softmax(logits, dim=-1)
            
            # 5. Robust Label Mapping
            # For 'padmalcom' model: 
            # Label 0 = "real" (HUMAN)
            # Label 1 = "fake" (AI)
            # We explicitly check this mapping to be safe.
            id2label = self.model.config.id2label
            fake_score = 0.0
            real_score = 0.0
            
            # Find which index is 'fake' and which is 'real'
            for idx, label in id2label.items():
                if "fake" in label.lower() or "spoof" in label.lower():
                    fake_score = probs[0][idx].item()
                elif "real" in label.lower() or "bonafide" in label.lower():
                    real_score = probs[0][idx].item()
            
            print(f"🔍 DEBUG: Real Score: {real_score:.4f} | Fake Score: {fake_score:.4f}")

            # 6. Decision Logic (Using a stricter threshold)
            # If AI confidence is > 55%, call it AI. Otherwise Human.
            if fake_score > 0.55:
                classification = "AI_GENERATED"
                confidence = fake_score
                explanation = f"Detected synthetic artifacts with {int(fake_score*100)}% confidence."
            else:
                classification = "HUMAN"
                confidence = real_score
                explanation = f"Verified human vocal characteristics with {int(real_score*100)}% confidence."

            return {
                "classification": classification,
                "confidenceScore": round(confidence, 2),
                "explanation": explanation
            }

        except Exception as e:
            print(f"Analysis Error: {e}")
            return {
                "classification": "HUMAN",
                "confidenceScore": 0.0,
                "explanation": f"Error: {str(e)}"
            }

# Helper to fix numpy integer issue in newer versions
def int_(val):
    return int(val)