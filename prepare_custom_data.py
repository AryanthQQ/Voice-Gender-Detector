import os
import csv
import ssl
import tempfile
import urllib.request
from main import extract_features, FEATURES

def process_custom_data(csv_path: str, output_csv: str, limit: int = None):
    print(f"Reading from: {csv_path}")
    print(f"Output will be saved to: {output_csv}")
    
    # Bypass SSL verification if needed
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    rows_to_process = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['voiceNoteUrl'].strip():
                rows_to_process.append(row)
    
    print(f"Total rows found: {len(rows_to_process)}")
    if limit is not None:
        rows_to_process = rows_to_process[:limit]
        print(f"Limiting processing to {limit} rows for testing.")

    # Prepare output CSV
    fieldnames = FEATURES + ['label']
    file_exists = os.path.exists(output_csv)
    
    processed_count = 0
    error_count = 0
    
    with open(output_csv, 'a', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        
        for i, row in enumerate(rows_to_process, 1):
            url = row['voiceNoteUrl'].strip()
            # In your data, gender is 'f' or 'm'. We map this to 'female' or 'male'
            gender = row.get('gender', '').strip().lower()
            label = 'female' if gender == 'f' else 'male' if gender == 'm' else None
            
            if not label:
                print(f"[{i}/{len(rows_to_process)}] Skipping row with unknown gender: {gender}")
                continue

            print(f"[{i}/{len(rows_to_process)}] Processing ({label}): {url}")
            
            try:
                # 1. Download file to temp
                req_obj = urllib.request.Request(
                    url,
                    headers={"User-Agent": "VoiceGenderDataPrep/1.0"}
                )
                with urllib.request.urlopen(req_obj, timeout=30, context=ctx) as resp:
                    content = resp.read()
                
                # Determine extension
                ext = os.path.splitext(url.split("?")[0])[1]
                if not ext: ext = ".mp3"
                if b'ftyp' in content[:16]:
                    ext = ".m4a"
                
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                
                try:
                    # 2. Extract features
                    features = extract_features(tmp_path)
                    
                    # 3. Save to CSV
                    features['label'] = label
                    writer.writerow(features)
                    out_f.flush()
                    processed_count += 1
                finally:
                    # Cleanup
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            
            except Exception as e:
                print(f"  -> Error processing {url}: {e}")
                error_count += 1

    print("\n--- Processing Complete ---")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare custom data for voice gender training")
    parser.add_argument("--input", default=r"c:\Users\hp\Downloads\Independent-Advisor-Voice-Notes.csv", help="Input CSV file")
    parser.add_argument("--output", default="custom_voice.csv", help="Output CSV file with extracted features")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N files (for testing)")
    
    args = parser.parse_args()
    process_custom_data(args.input, args.output, args.limit)
