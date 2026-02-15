import json
import numpy as np
import torch
import librosa
import os
import io
import traceback
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
from utils import decode_base64_audio, convert_mp3_to_wav
from detector import VoiceDetector
from scipy.io.wavfile import write

app = FastAPI(title="Sentient Guard: Emotion & Scam API")

print("⏳ Waking up the AI... (Loading 4 Models)")

# --- 1. THE AI BRAIN (4-in-1) ---

# A. Voice Authenticity (Is it Human?)
voice_detector = VoiceDetector()

# B. Ears (Transcription)
transcriber = WhisperModel("tiny", device="cpu", compute_type="int8")

# C. Semantic Brain (Scam Pattern Matching)
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# D. The "Heart" (Emotion & Feeling Understanding)
# Detects: joy, sadness, anger, fear, surprise, neutral
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

print("✅ AI is Awake and Feeling.")

# --- KNOWLEDGE BASE ---
SCAM_KNOWLEDGE_BASE = [
    "Share the one time password sent to your phone",
    "Verify your bank account details immediately",
    "Your credit card has been blocked due to suspicious activity",
    "This is a call from the police department regarding a warrant",
    "You have won a lottery, pay tax to claim it",
    "Download AnyDesk or TeamViewer for remote support",
    "Pay the customs duty for your parcel",
    "Your electricity will be disconnected tonight",
    "Click the link sent via SMS to update KYC",
    "Your child is kidnapped send money now"
]

# Pre-calculate scam concepts
kb_embeddings = semantic_model.encode(SCAM_KNOWLEDGE_BASE, convert_to_tensor=True)

SAFE_CONTEXTS = ["hackathon", "project", "demo", "test", "movie", "play", "acting", "drama"]

# --- API CONFIGURATION ---
API_KEY = os.getenv("API_KEY", "sk_test_123456789")

# --- REQUEST MODEL ---
class DetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# --- SINGLE REST ENDPOINT (STRICTLY COMPLIANT) ---
@app.post("/api/voice-detection")
async def detect_voice(request: DetectionRequest, x_api_key: str = Header(None)):
    """
    Single endpoint with AI detection, transcription, emotion analysis, and scam detection.
    Strictly follows Hackathon PDF requirements for response format.
    """
    
    # --- AUTHENTICATION ---
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    try:
        # --- STEP 1: Decode and Convert Audio ---
        mp3 = decode_base64_audio(request.audioBase64)
        wav = convert_mp3_to_wav(mp3)
        
        # --- STEP 2: Check for AI Voice (Voice Authenticity) ---
        voice_result = voice_detector.analyze(wav, request.language)
        
        # Ensure strict types for PDF compliance
        classification = voice_result.get("classification", "HUMAN") # Must be "HUMAN" or "AI_GENERATED"
        confidence_score = float(voice_result.get("confidenceScore", 0.0)) # Must be float 0.0-1.0
        
        # --- STEP 3: Transcribe Audio ---
        wav.seek(0)
        audio_input, sr = librosa.load(wav, sr=16000)
        transcript_text = ""
        try:
            segments, _ = transcriber.transcribe(audio_input, beam_size=1)
            for segment in segments:
                transcript_text += segment.text + " "
        except Exception as e:
            print(f"Transcription Error: {e}")
        
        # --- STEP 4: Emotion Analysis ---
        detected_emotion = "neutral"
        emotion_score = 0.0
        triggers = []
        
        if transcript_text.strip():
            transcript_lower = transcript_text.lower()
            
            try:
                # Analyze emotion from text
                emotions = emotion_classifier(transcript_text)
                detected_emotion = emotions[0][0]['label']
                emotion_score = emotions[0][0]['score']
                
                # LOGIC: Scams often use FEAR (threats) or SURPRISE (lottery)
                if detected_emotion in ["fear", "anger"] and emotion_score > 0.6:
                    triggers.append(f"High {detected_emotion.upper()} detected ({int(emotion_score*100)}%)")
                elif detected_emotion == "surprise" and "win" in transcript_lower:
                    triggers.append("Suspicious Surprise (Lottery scam?)")
                    
            except Exception as e:
                print(f"Emotion Error: {e}")
            
            # --- STEP 5: Semantic Scam Detection ---
            is_safe = any(word in transcript_lower for word in SAFE_CONTEXTS)
            
            if not is_safe:
                try:
                    user_embedding = semantic_model.encode(transcript_text, convert_to_tensor=True)
                    cosine_scores = util.cos_sim(user_embedding, kb_embeddings)
                    best_score = cosine_scores[0][torch.argmax(cosine_scores).item()].item()
                    
                    if best_score > 0.55:  # 55% similarity
                        triggers.append(f"Scam Pattern Match ({int(best_score*100)}%)")
                except Exception as e:
                    print(f"Semantic Error: {e}")
        
        # --- STEP 6: Calculate Risk Score ---
        current_risk = 0
        
        if triggers:
            current_risk = int(len(triggers) * 25)
        
        # AI Voice is always high risk
        if classification == "AI_GENERATED" and confidence_score > 0.8:
            current_risk = 100
            triggers.append(f"AI Voice Detected ({int(confidence_score*100)}%)")
        
        # Determine Alert Status
        alert_status = "SAFE"
        if current_risk > 80:
            alert_status = "CRITICAL_THREAT"
        elif current_risk > 50:
            alert_status = "DANGER_HIGH"
        elif current_risk > 20:
            alert_status = "WARNING_SUSPICIOUS"
        
        # --- RETURN RESPONSE (STRICT FORMAT FOR EVALUATOR) ---
        response = {
            # === STRICT PDF REQUIREMENTS ===
            "status": "success",
            "classification": classification,
            "confidenceScore": confidence_score,
            
            # === YOUR ADVANCED FEATURES ===
            "alert": alert_status,
            "risk_score": min(100, current_risk),
            "transcript": transcript_text.strip(),
            "emotion": detected_emotion,
            "triggers": triggers,
            "spam": "Yes" if current_risk > 50 else "No"
        }
        
        return response
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}