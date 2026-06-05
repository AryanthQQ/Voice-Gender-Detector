"""
test_predict_url_debug.py — Debug 500 error
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

try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
        print("SUCCESS!", result)
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}")
    print("Detail:", e.read().decode())
except Exception as e:
    print(f"Error: {e}")
