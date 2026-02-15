import requests
import base64
import json
import os
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION ---
ENDPOINT_URL = 'https://testing-ak-voice-detection-v2.hf.space/api/voice-detection'
API_KEY = 'sk_test_123456789'

def create_session_with_retries():
    """Create a session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def test_audio_file(filepath, session):
    """Test a single audio file"""
    filename = Path(filepath).name
    print(f"\n{'='*60}")
    print(f"🎵 Testing: {filename}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return None

    try:
        # 1. Read Audio
        with open(filepath, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')

        # 2. Prepare Payload
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
        response = session.post(ENDPOINT_URL, json=payload, headers=headers, timeout=(30, 120))
        
        # 4. Validate Response
        if response.status_code == 200:
            data = response.json()
            print("✅ HTTP 200 OK")
            
            # Check Required Fields
            if "status" in data and "classification" in data and "confidenceScore" in data:
                print("✅ JSON Structure Valid")
                
                status = data.get("status")
                classification = data.get("classification")
                score = data.get("confidenceScore")
                
                print(f"   Status: {status}")
                print(f"   Classification: {classification}")
                print(f"   Confidence Score: {score}")
                
                return {
                    "filename": filename,
                    "status": "success",
                    "classification": classification,
                    "confidence": score,
                    "http_code": 200
                }
            else:
                print("❌ Missing required fields!")
                print(f"   Received keys: {list(data.keys())}")
                return {
                    "filename": filename,
                    "status": "error",
                    "error": "Missing fields"
                }
        else:
            print(f"❌ HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return {
                "filename": filename,
                "status": "error",
                "error": f"HTTP {response.status_code}"
            }

    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout Error: {e}")
        return {
            "filename": filename,
            "status": "error",
            "error": f"Timeout"
        }
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        return {
            "filename": filename,
            "status": "error",
            "error": "Connection Error"
        }
    except Exception as e:
        print(f"❌ Error: {e}")
        return {
            "filename": filename,
            "status": "error",
            "error": str(e)
        }

def main():
    print(f"{'='*60}")
    print(f"🚀 TESTING ALL AUDIO FILES")
    print(f"Target: {ENDPOINT_URL}")
    print(f"{'='*60}")
    
    # Find all MP3 files
    mp3_files = sorted(Path(".").glob("*.mp3"))
    
    if not mp3_files:
        print("⚠️ No MP3 files found in the current directory!")
        return
    
    print(f"\n📁 Found {len(mp3_files)} audio file(s):\n")
    for i, f in enumerate(mp3_files, 1):
        print(f"   {i}. {f.name}")
    
    # Create session with retry logic
    session = create_session_with_retries()
    
    results = []
    for filepath in mp3_files:
        result = test_audio_file(str(filepath), session)
        if result:
            results.append(result)
    
    # Close session
    session.close()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total files tested: {len(results)}")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    print(f"✅ Successful: {len(successful)}")
    print(f"❌ Failed: {len(failed)}")
    
    if successful:
        print(f"\n🎉 Successful Tests:")
        for r in successful:
            print(f"   • {r['filename']}")
            print(f"     └─ {r['classification']} (confidence: {r['confidence']})")
    
    if failed:
        print(f"\n⚠️ Failed Tests:")
        for r in failed:
            print(f"   • {r['filename']}")
            print(f"     └─ {r['error']}")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
