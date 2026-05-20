import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/processed/chunks.json"
OUTPUT_INDEX = "data/processed/faiss.index"


# LOAD CHUNKS

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    chunks = json.load(f)


print("Loaded chunks:", len(chunks))



# LOAD EMBEDDING MODEL

model = SentenceTransformer(
    "keepitreal/vietnamese-sbert"
)


# CREATE EMBEDDINGS

embeddings = model.encode(
    chunks,
    normalize_embeddings=True,
    show_progress_bar=True
)


embeddings = np.array(
    embeddings,
    dtype="float32"
)


print("Embedding shape:", embeddings.shape)


# CREATE FAISS INDEX

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)


# SAVE

faiss.write_index(
    index,
    OUTPUT_INDEX
)


print("=" * 50)
print("✅ FAISS BUILD DONE")
print("✅ Total vectors:", index.ntotal)
print("✅ Saved:", OUTPUT_INDEX)
print("=" * 50)