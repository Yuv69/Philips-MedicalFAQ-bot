# import numpy as np
# import faiss

# # Load embeddings
# embeddings = np.load(r'C:\Users\yuvra\OneDrive\Desktop\philips.data\faq_embeddings.npy')

# # Ensure correct format
# embeddings = embeddings.astype('float32')

# # Build FAISS index
# index = faiss.IndexFlatL2(embeddings.shape[1])
# index.add(embeddings)

# # Save index
# faiss.write_index(index, r'C:\Users\yuvra\OneDrive\Desktop\philips.data\faiss_index.bin')
# print(f"FAISS index built with {embeddings.shape[0]} vectors.")




import numpy as np
import faiss
import os
import hashlib

DATA_PATH = r'C:\Users\yuvra\OneDrive\Desktop\philips.data'
EMBEDDINGS_FILE = os.path.join(DATA_PATH, 'faq_embeddings.npy')
INDEX_FILE = os.path.join(DATA_PATH, 'faiss_index.bin')

def build_index():
    # Load embeddings with validation
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Embeddings file not found. Run generate_embeddings.py first.")
        return

    try:
        embeddings = np.load(EMBEDDINGS_FILE).astype('float32')
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        return

    # Validate embeddings
    if len(embeddings.shape) != 2:
        print(f"Invalid embeddings shape: {embeddings.shape}")
        return

    print(f"Building index for {embeddings.shape[0]} vectors...")
    
    # Create optimized index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index with verification
    faiss.write_index(index, INDEX_FILE)
    print(f"Index saved to {INDEX_FILE}")
    
    # Add size verification
    index_size = os.path.getsize(INDEX_FILE)
    print(f"Index file size: {index_size/1024/1024:.2f} MB")
    
    # Add checksum verification
    with open(INDEX_FILE, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    print(f"MD5 checksum: {file_hash}")

if __name__ == "__main__":
    build_index()









