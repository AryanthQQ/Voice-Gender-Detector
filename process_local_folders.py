import os
import csv
import sys

# Add path to import main
sys.path.append(r'c:\Users\hp\Downloads\voice-gender-master\voice-gender-master')

import os
# Add current dir to PATH so audioread finds ffmpeg.exe
os.environ["PATH"] += os.pathsep + os.path.abspath(os.path.dirname(__file__))

from main import extract_features, FEATURES

def process_folders(male_dir, female_dir, output_csv):
    fieldnames = FEATURES + ['label']
    
    male_files = [os.path.join(male_dir, f) for f in os.listdir(male_dir) if f.endswith(('.wav', '.mp3', '.m4a'))]
    female_files = [os.path.join(female_dir, f) for f in os.listdir(female_dir) if f.endswith(('.wav', '.mp3', '.m4a'))]
    
    all_files = [(f, 'male') for f in male_files] + [(f, 'female') for f in female_files]
    
    print(f"Found {len(male_files)} male files and {len(female_files)} female files.")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        
        success = 0
        errors = 0
        
        for i, (filepath, label) in enumerate(all_files, 1):
            print(f"[{i}/{len(all_files)}] Processing {label}: {os.path.basename(filepath)}")
            try:
                features = extract_features(filepath)
                features['label'] = label
                writer.writerow(features)
                success += 1
            except Exception as e:
                print(f"  -> Error: {e}")
                errors += 1
                
    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {success}")
    print(f"Errors: {errors}")

if __name__ == "__main__":
    male_dir = r"C:\Users\hp\Desktop\male voice"
    female_dir = r"C:\Users\hp\Desktop\Female voice"
    output_csv = r"C:\Users\hp\Downloads\voice-gender-master\voice-gender-master\custom_voice.csv"
    process_folders(male_dir, female_dir, output_csv)
