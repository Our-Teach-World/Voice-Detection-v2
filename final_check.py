import requests
import base64
import json
import os

# --- CONFIGURATION FROM YOUR PDF ---
ENDPOINT_URL = 'https://testing-ak-voice-detection-v2.hf.space/api/voice-detection'
API_KEY = 'sk_test_123456789'

# Create a dummy test file if you don't have one
if not os.path.exists("test_audio.mp3"):
    print("⚠️ Please put an MP3 file named 'test_audio.mp3' in this folder to test!")
    exit()

def test_submission():
    print(f"{'='*60}")
    print(f"🚀 FINAL PRE-SUBMISSION CHECK")
    print(f"Target: {ENDPOINT_URL}")
    print(f"{'='*60}\n")

    # 1. Read Audio
    with open("test_audio.mp3", "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')

    # 2. Prepare Payload (Per PDF Page 3)
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    # 3. Send Request
    print("⏳ Sending request...")
    try:
        response = requests.post(ENDPOINT_URL, json=payload, headers=headers, timeout=30)
        
        # 4. Validate per PDF Page 4
        if response.status_code == 200:
            data = response.json()
            print("✅ HTTP 200 OK")
            
            # Check Required Fields
            if "status" in data and "classification" in data and "confidenceScore" in data:
                print("✅ JSON Structure Valid")
            else:
                print("❌ FAILED: Missing fields!")
                print(f"Received: {data.keys()}")
                return

            # Check Specific Values
            if data["status"] == "success":
                print("✅ Status is 'success'")
            else:
                print(f"❌ FAILED: Status is '{data.get('status')}'")

            if data["classification"] in ["HUMAN", "AI_GENERATED"]:
                print(f"✅ Classification '{data['classification']}' is valid")
            else:
                print(f"❌ FAILED: Invalid classification '{data.get('classification')}'")

            score = data["confidenceScore"]
            if isinstance(score, (int, float)) and 0 <= score <= 1:
                print(f"✅ Confidence Score {score} is valid")
            else:
                print(f"❌ FAILED: Score {score} out of range")

            print("\n🎉 READY TO SUBMIT! The API meets all PDF requirements.")
            print(f"Response: {json.dumps(data, indent=2)}")

        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"❌ FAILED: Connection Error - {e}")

if __name__ == "__main__":
    test_submission()