import pandas as pd
import glob
import json
import os

# Define the directory path where your CSV files are located
data_dir = r'C:\Users\yuvra\OneDrive\Desktop\philips.data'

# Change the working directory to that path
os.chdir(data_dir)

# Get all CSV files in the directory
csv_files = glob.glob('*.csv')
faq_list = []

for file in csv_files:
    try:
        df = pd.read_csv(file)
        # Print found columns for debugging
        print(f"Reading {file}: columns found: {df.columns.tolist()}")

        # Look for the correct columns, case-insensitive
        col_map = {c.lower(): c for c in df.columns}

        if 'question' in col_map and 'answer' in col_map:
            question_col = col_map['question']
            answer_col = col_map['answer']

            for _, row in df.iterrows():
                q = str(row[question_col]).strip()
                a = str(row[answer_col]).strip()
                if q and a:
                    faq_list.append({"question": q, "answer": a})
        else:
            print(f"Skipping {file}: 'Question' and/or 'Answer' column missing.")
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not faq_list:
    raise ValueError("No FAQ data found in the provided CSV files.")

# Save the JSON in the same directory
json_path = os.path.join(data_dir, 'merged_faq.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(faq_list, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(faq_list)} Q&A pairs from {len(csv_files)} CSV file(s).")
print(f"Saved to: {json_path}")




