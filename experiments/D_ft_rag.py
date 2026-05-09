import json
import re
from pathlib import Path

import faiss
import numpy as np
import torch

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# =====================================================
# PATHS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"

FAISS_PATH = ROOT_DIR / "data" / "processed" / "faiss.index"

FT_MODEL = ROOT_DIR / "finetuned_model"


# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)


# =====================================================
# LOAD CHUNKS
# =====================================================

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:

    chunks = json.load(f)

print("Chunks:", len(chunks))


# =====================================================
# LOAD FAISS
# =====================================================

index = faiss.read_index(
    str(FAISS_PATH)
)

print("FAISS:", index.ntotal)


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

embedding_model = SentenceTransformer(
    "keepitreal/vietnamese-sbert",
    device=DEVICE
)


# =====================================================
# LOAD BASE MODEL
# =====================================================

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
).to(DEVICE)


# =====================================================
# LOAD FINETUNED MODEL
# =====================================================

model = PeftModel.from_pretrained(
    base_model,
    str(FT_MODEL)
)

model.eval()

print("Fine-tuned model loaded")


# =====================================================
# CLEAN CONTEXT
# =====================================================

def clean_context(text):

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text


# =====================================================
# RETRIEVE
# =====================================================

def retrieve(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for idx in indices[0]:

        if idx < len(chunks):

            results.append(
                chunks[idx]
            )

    return results


# =====================================================
# SIMPLE EXTRACTION
# =====================================================

def extract_year(context):

    years = re.findall(
        r"(\\d{3,4})",
        context
    )

    if years:

        return years[0]

    return None


# =====================================================
# ASK FUNCTION
# =====================================================

def ask_D(question):

    retrieved = retrieve(
        question,
        top_k=1
    )


    context = clean_context(
    "\n\n".join(retrieved[:3])
    )


    # =============================================
    # SIMPLE FACT EXTRACTION
    # =============================================

    if "năm" in question.lower():

        year = extract_year(context)

        if year:

            return year


    # =============================================
    # LLM
    # =============================================

    prompt = f"""
Bạn là trợ lý lịch sử Việt Nam.

Chỉ trả lời bằng thông tin có trong context.

Nếu context không chứa đáp án thì trả lời:
"Không tìm thấy thông tin."

Trả lời thật ngắn gọn.

Context:
{context}

Question:
{question}

Answer:
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            do_sample=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            temperature=0.0,
            max_new_tokens=8,
            pad_token_id=tokenizer.eos_token_id
        )


    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


    if "assistant:" in answer:

        answer = answer.split(
            "assistant:"
        )[-1]


    # remove extra artifacts
    answer = answer.replace(
        "Đúng,",
        ""
    )

    answer = answer.strip()

    return answer


# =====================================================
# CHAT LOOP
# =====================================================

if __name__ == "__main__":

    while True:

        question = input("\nQuestion: ")

        answer = ask_D(question)

        print("\nAnswer:", answer)