# =========================================================
# FILE: rag/build_db.py
# =========================================================

import json
import re

from pathlib import Path
from underthesea import sent_tokenize


INPUT_FILE = "data/raw/history_clean.txt"

OUTPUT_FILE = "data/processed/chunks.json"


# =========================================================
# LOAD
# =========================================================

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    text = f.read()


# =========================================================
# CLEAN
# =========================================================

# remove wiki tags
text = re.sub(r"\[[^\]]*\]", " ", text)

# remove chinese chars
text = re.sub(r"[㐀-䶿一-鿿]", " ", text)

# remove extra spaces
text = re.sub(r"\s+", " ", text)

# remove spaces before punctuation
text = re.sub(r"\s+([,.!?;:])", r"\1", text)

# normalize ...
text = text.replace("...", " ")

text = text.strip()


# =========================================================
# PARAGRAPH SPLIT
# =========================================================

paragraphs = re.split(r'\n+', text)


all_chunks = []


# =========================================================
# PROCESS EACH PARAGRAPH
# =========================================================

for para in paragraphs:

    para = para.strip()

    if len(para) < 80:
        continue


    # sentence tokenize
    sentences = sent_tokenize(para)


    # clean sentences
    clean_sentences = []


    for s in sentences:

        s = s.strip()

        if len(s) < 20:
            continue

        clean_sentences.append(s)


    # =====================================================
    # MERGE SEMANTIC SENTENCES
    # =====================================================

    temp = ""


    for s in clean_sentences:

        # merge until ~350 chars
        if len(temp) + len(s) < 220:

            temp += " " + s

        else:

            if 80 <= len(temp) <= 500:

                all_chunks.append(temp.strip())

            temp = s


    # remain
    if 80 <= len(temp) <= 500:

        all_chunks.append(temp.strip())


# =========================================================
# OVERLAP CHUNKING
# =========================================================

final_chunks = []


for i in range(len(all_chunks)):

    current = all_chunks[i]

    final_chunks.append(current)


    # overlap with next
    if i < len(all_chunks) - 1:

        overlap = current + " " + all_chunks[i + 1]

        if len(overlap) < 700:

            final_chunks.append(overlap)


# =========================================================
# FILTER
# =========================================================

blacklist = [
    "tham khảo",
    "isbn",
    "liên kết ngoài",
    "archive",
    "truy cập ngày"
]


filtered = []


for c in final_chunks:

    lower = c.lower()

    skip = False


    for word in blacklist:

        if word in lower:

            skip = True
            break


    if skip:
        continue


    # clean spaces again
    c = re.sub(r"\s+", " ", c)

    c = c.strip()


    filtered.append(c)


# =========================================================
# REMOVE DUPLICATES
# =========================================================

filtered = list(dict.fromkeys(filtered))


# =========================================================
# SAVE
# =========================================================

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        filtered,
        f,
        ensure_ascii=False,
        indent=2
    )


# =========================================================
# STATS
# =========================================================

print("=" * 50)

print("✅ SEMANTIC HYBRID CHUNKING DONE")

print("✅ Total chunks:", len(filtered))

print("✅ Saved:", OUTPUT_FILE)

print("=" * 50)