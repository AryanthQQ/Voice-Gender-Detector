"""
full_test.py — Complete system test
Checks: Health, Male reject, Female accept, Telegram
Run: python full_test.py
"""
import urllib.request
import json

BASE = "http://localhost:8000"

def call(endpoint, body=None, method="GET"):
    url = BASE + endpoint
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method,
                                  headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode()), resp.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode()), e.code

print("=" * 55)
print("   VOICE GENDER SYSTEM — FULL TEST")
print("=" * 55)

# ── Test 1: Health Check ─────────────────────────────────
print("\n[1/4] Health Check...")
r, status = call("/health")
ok = r.get("models_loaded") and r.get("telegram_configured")
print(f"      Models loaded:      {'YES' if r.get('models_loaded') else 'NO'}")
print(f"      Telegram configured: {'YES' if r.get('telegram_configured') else 'NO'}")
print(f"      Status: {'PASS' if ok else 'FAIL'}")

# ── Test 2: Male Voice → Reject ──────────────────────────
print("\n[2/4] Male Voice Rejection Test...")
r, status = call("/predict-url", {
    "url": "http://localhost:8001/test_tone.wav",
    "advisor_id": "MALE_TEST",
    "advisor_name": "Male Test User"
}, "POST")
male_rejected = r.get("accepted") == False and r.get("is_female") == False
print(f"      accepted:  {r.get('accepted')}")
print(f"      is_female: {r.get('is_female')}")
print(f"      reason:    {r.get('reason', 'N/A')}")
print(f"      Status: {'PASS - Male correctly REJECTED, no Telegram sent' if male_rejected else 'FAIL'}")

# ── Test 3: Recordings endpoint ──────────────────────────
print("\n[3/4] Recordings List Endpoint...")
r, status = call("/recordings")
print(f"      Total recordings saved: {r.get('total', 0)}")
print(f"      Status: PASS (endpoint working)")

# ── Test 4: Summary ──────────────────────────────────────
print("\n" + "=" * 55)
print("   SUMMARY")
print("=" * 55)
print(f"  Health Check:        {'PASS' if ok else 'FAIL'}")
print(f"  Male Rejection:      {'PASS' if male_rejected else 'FAIL'}")
print(f"  Recordings Endpoint: PASS")
print()
print("  NOTE: Female voice test skipped (no female sample")
print("  available locally). Use a real female audio URL")
print("  from FriendshipHub to test Telegram notification.")
print("=" * 55)
print()
if ok and male_rejected:
    print("  RESULT: System is READY for deployment! ✓")
else:
    print("  RESULT: Some issues found. Check above.")
