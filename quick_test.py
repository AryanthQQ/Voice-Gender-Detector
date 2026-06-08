import threading
import time
import urllib.request
import json
import uvicorn
from main import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(3) # wait for server to start

body = json.dumps({
    "url": "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
    "advisor_id": "TEST001",
    "advisor_name": "Test Advisor"
}).encode()

req = urllib.request.Request(
    "http://127.0.0.1:8000/predict-url",
    data=body,
    method="POST",
    headers={"Content-Type": "application/json"}
)

print("Server started. Sending request for S3 Audio...")
try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read().decode())
        print("\n=== FINAL RESULT ===")
        print(json.dumps(result, indent=2))
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}")
    print("Detail:", e.read().decode())
except Exception as e:
    print(f"Error: {e}")
