# import json
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Load FAQ data
# with open(r'C:\Users\yuvra\OneDrive\Desktop\philips.data\merged_faq.json', 'r', encoding='utf-8') as f:
#     faq_list = json.load(f)

# # Initialize list to hold combined question and answer
# texts = []

# # Ensure 'question' and 'answer' keys exist in the items
# for item in faq_list:
#     if 'question' in item and 'answer' in item:
#         texts.append(f"{item['question']} {item['answer']}")
#     else:
#         print(f"Missing 'question' or 'answer' in item: {item}")

# # Check if texts list is empty
# if not texts:
#     print("No valid texts to generate embeddings from.")
# else:
#     # Generate embeddings
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     embeddings = model.encode(texts, show_progress_bar=True)

#     # Save embeddings to the same folder
#     np.save(r'C:\Users\yuvra\OneDrive\Desktop\philips.data\faq_embeddings.npy', embeddings)
#     print(f"Embeddings shape: {embeddings.shape}")

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import hashlib
import os

DATA_PATH = r'C:\Users\yuvra\OneDrive\Desktop\philips.data'
INPUT_FILE = os.path.join(DATA_PATH, 'merged_faq.json')
OUTPUT_FILE = os.path.join(DATA_PATH, 'faq_embeddings.npy')

def generate_embeddings():
    # Load and validate FAQ data
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            faq_list = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        return

    # Process data with error handling
    texts = []
    for idx, item in enumerate(faq_list):
        try:
            if 'question' in item and 'answer' in item:
                texts.append(f"{item['question']} {item['answer']}")
            elif 'intents' in item:
                for intent in item.get('intents', []):
                    patterns = intent.get('patterns', [])
                    responses = intent.get('responses', [])
                    for pattern in patterns:
                        texts.append(pattern)
                        for response in responses:
                            texts.append(response)
        except KeyError as e:
            print(f"Skipping item {idx} due to missing key: {str(e)}")
            continue

    if not texts:
        print("No valid text data found for embedding generation")
        return

    # Generate embeddings with progress tracking
    print(f"Generating embeddings for {len(texts)} text items...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Save with verification
    np.save(OUTPUT_FILE, embeddings)
    print(f"Embeddings saved to {OUTPUT_FILE}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Add checksum verification
    with open(OUTPUT_FILE, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    print(f"MD5 checksum: {file_hash}")

if __name__ == "__main__":
    generate_embeddings()










