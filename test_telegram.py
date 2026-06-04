"""
test_telegram.py — Test Telegram notification independently
Run: python test_telegram.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import config

print("=" * 50)
print("Telegram Notification Test")
print("=" * 50)
print(f"Bot Token: {'[SET]' if config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_BOT_TOKEN != 'YOUR_BOT_TOKEN_HERE' else '[NOT SET - edit .env]'}")
print(f"Chat ID:   {'[SET]' if config.TELEGRAM_CHAT_ID and config.TELEGRAM_CHAT_ID != 'YOUR_CHAT_ID_HERE' else '[NOT SET - edit .env]'}")
print(f"Notify on: {config.NOTIFY_ON}")
print()

if not config.telegram_configured():
    print("ERROR: Telegram is not configured!")
    print()
    print("Steps to configure:")
    print("  1. Open Telegram -> search @BotFather -> /newbot -> copy token")
    print("  2. Search @userinfobot -> start -> copy your Chat ID (number)")
    print("  3. Edit the .env file in this folder:")
    print("     TELEGRAM_BOT_TOKEN=123456789:ABCdef...")
    print("     TELEGRAM_CHAT_ID=123456789")
    print()
    print("Then run this script again.")
    sys.exit(1)

# Send test notification
import urllib.request, urllib.parse, json

print("Sending test notification...")

# Dummy result for testing
dummy_result = {
    'svm':      {'label': 'female', 'confidence': 89.3},
    'gbm':      {'label': 'female', 'confidence': 94.1},
    'rf':       {'label': 'female', 'confidence': 82.5},
    'ensemble': {'label': 'female', 'confidence': 88.6, 'male_votes': 0, 'total_votes': 3},
    'features': {'meanfun_hz': 198.0, 'meanfreq_hz': 212.0, 'IQR': 0.0891},
}

from datetime import datetime
now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

msg = (
    f"🎙️ <b>Voice Gender Verification Alert</b>\n"
    f"━━━━━━━━━━━━━━━━━━━━━━\n"
    f"📅 <b>Time:</b> {now}\n"
    f"🔊 <b>File:</b> <code>test_message.wav</code>\n"
    f"📁 <b>Size:</b> 96.0 KB\n\n"
    f"✅ 👩 VERDICT: <b>FEMALE VERIFIED</b>\n"
    f"<b>Confidence:</b> 88.6%\n"
    f"━━━━━━━━━━━━━━━━━━━━━━\n"
    f"<b>Model Breakdown:</b>\n"
    f"  • SVM:            Female (89%)\n"
    f"  • Gradient Boost: Female (94%)\n"
    f"  • Random Forest:  Female (83%)\n"
    f"  • Ensemble Vote:  3/3 Female\n\n"
    f"<b>Voice Analysis:</b>\n"
    f"  • Avg Fundamental Freq: 198 Hz\n"
    f"  • Mean Frequency:       212 Hz\n"
    f"  • Variability (IQR):    0.0891\n"
    f"━━━━━━━━━━━━━━━━━━━━━━\n"
    f"<i>Auto-verified by Voice Gender AI v2.0</i>\n\n"
    f"<b>[THIS IS A TEST MESSAGE]</b>"
)

base = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
payload = urllib.parse.urlencode({
    'chat_id':    config.TELEGRAM_CHAT_ID,
    'text':       msg,
    'parse_mode': 'HTML',
}).encode()

req = urllib.request.Request(base, data=payload, method='POST')
req.add_header('Content-Type', 'application/x-www-form-urlencoded')

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read().decode())
        if result.get('ok'):
            print("SUCCESS! Check your Telegram — test message sent!")
        else:
            print(f"Telegram API error: {result}")
except Exception as e:
    print(f"Failed to send: {e}")
    print()
    print("Possible issues:")
    print("  - Bot token is wrong")
    print("  - Chat ID is wrong")
    print("  - No internet connection")
    print("  - Bot has not been started (send /start to your bot first)")
