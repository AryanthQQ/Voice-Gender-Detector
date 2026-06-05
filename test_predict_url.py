"""
test_predict_url.py — Test /predict-url endpoint
Run: python test_predict_url.py
"""
import urllib.request
import json

body = json.dumps({
    "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "advisor_id": "TEST001",
    "advisor_name": "Test Advisor"
}).encode()

req = urllib.request.Request(
    "http://localhost:8000/predict-url",
    data=body,
    method="POST",
    headers={"Content-Type": "application/json"}
)

print("Testing /predict-url endpoint...")
print("Downloading audio + analyzing (may take 30-60 sec)...")
print()

try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
        print("=" * 50)
        print("SUCCESS!")
        print("=" * 50)
        ens = result["ensemble"]
        print(f"Gender:      {ens['label'].upper()}")
        print(f"Confidence:  {ens['confidence']:.1f}%")
        print(f"Is Female:   {result['is_female']}")
        print(f"Source URL:  {result['source_url']}")
        print(f"Telegram:    {result['telegram_configured']}")
        print()
        print("Check your Telegram — notification with clickable audio link should be there!")
except Exception as e:
    print(f"Error: {e}")
