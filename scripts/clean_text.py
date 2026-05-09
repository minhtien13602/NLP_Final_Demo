# =========================================================
# FILE: scripts/clean_text.py
# =========================================================

import os
import re


RAW_DIR = "data/raw"

OUTPUT_FILE = "data/raw/history_clean.txt"


all_text = ""


for filename in os.listdir(RAW_DIR):

    if not filename.endswith(".txt"):
        continue

    path = os.path.join(RAW_DIR, filename)

    with open(path, "r", encoding="utf-8") as f:

        text = f.read()

    # remove all wikipedia tags
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # remove weird chars
    text = re.sub(r"[\xa0\t]", " ", text)

    all_text += text + "\n"


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(all_text)

print("Saved clean text")