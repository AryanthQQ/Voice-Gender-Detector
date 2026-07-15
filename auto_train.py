import os
import time
import schedule
from deepfake_detector_v2 import AdvancedDeepfakeDetector
from datetime import datetime

def job():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Auto-Training Start ho rahi hai...")
    try:
        detector = AdvancedDeepfakeDetector()
        
        # Ye function data/real aur data/fake folders se data uthayega aur model ko retrain karega
        detector.train_on_your_data(real_dir="data/real", fake_dir="data/fake")
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✅ Auto-Training Successfully Complete ho gayi!")
    except Exception as e:
        print(f"❌ Auto-Training mein error aaya: {e}")

# Yahan hum schedule set kar rahe hain
# Aap isko apni zaroorat ke hisaab se change kar sakte hain (e.g. har raat 2 baje ya har 12 ghante baad)
schedule.every().day.at("02:00").do(job)

if __name__ == "__main__":
    print("🚀 Auto-Train scheduler start ho gaya hai. Ab model apne aap schedule ke hisaab se train hoga.")
    print("Ensure karein ki naye audios 'data/real' ya 'data/fake' folder mein regularly daale jaa rahe hain.")
    
    # Pehli baar abhi train karne ke liye line uncomment karein:
    # job()

    while True:
        schedule.run_pending()
        time.sleep(60) # Har 1 minute mein check karega ki time hua hai ya nahi
