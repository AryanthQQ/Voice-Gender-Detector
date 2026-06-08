import csv

def combine_csvs(file1, file2, output_file):
    with open(file1, 'r', encoding='utf-8') as f1:
        reader1 = list(csv.reader(f1))
        
    with open(file2, 'r', encoding='utf-8') as f2:
        reader2 = list(csv.reader(f2))
        # Skip header for the second file
        reader2 = reader2[1:]
        
    combined = reader1 + reader2
    
    with open(output_file, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerows(combined)
        
    print(f"Combined {len(reader1)-1} + {len(reader2)} = {len(combined)-1} rows.")

combine_csvs('voice.csv', 'custom_voice.csv', 'combined_voice.csv')
