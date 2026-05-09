import json
import re
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
import torch

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Vietnam History QA",
    layout="wide"
)

st.title("Vietnam History QA RAG(Demo)")


# =====================================================
# PATHS
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = ROOT_DIR / "data" / "processed" / "chunks.json"

FAISS_PATH = ROOT_DIR / "data" / "processed" / "faiss.index"


MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("DEVICE:", DEVICE)


# =====================================================
# LOAD CHUNKS
# =====================================================

@st.cache_data
def load_chunks():

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:

        return json.load(f)


chunks = load_chunks()

print("Loaded chunks:", len(chunks))


# =====================================================
# LOAD FAISS
# =====================================================

@st.cache_resource
def load_faiss():

    return faiss.read_index(str(FAISS_PATH))


index = load_faiss()

print("FAISS total:", index.ntotal)


# =====================================================
# EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "keepitreal/vietnamese-sbert",
        device=DEVICE
    )


embedding_model = load_embedding_model()


# =====================================================
# LOAD LLM
# =====================================================

@st.cache_resource
def load_llm():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        device_map="auto"
    )

    return tokenizer, model


tokenizer, model = load_llm()


# =====================================================
# RETRIEVAL
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

            results.append(chunks[idx])

    return results


# =====================================================
# CLEAN ANSWER
# =====================================================

def clean_answer(text):

    text = re.sub(
        r"Question:.*",
        "",
        text,
        flags=re.DOTALL
    )

    text = re.sub(
        r"Answer:.*",
        "",
        text,
        flags=re.DOTALL
    )

    return text.strip()


# =====================================================
# GENERATE
# =====================================================

def generate_answer(question, context):

    prompt = f"""
Bạn là trợ lý lịch sử Việt Nam.

Chỉ được trả lời dựa trên context.

Nếu không có thông tin thì nói:

'Tôi không tìm thấy thông tin phù hợp.'

Context:
{context}

Câu hỏi:
{question}

Trả lời:
"""


    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)


    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id
        )


    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    answer = answer.replace(prompt, "")

    answer = clean_answer(answer)

    return answer


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("⚙️ Settings")

top_k = st.sidebar.slider(
    "Top K Retrieval",
    min_value=1,
    max_value=10,
    value=3
)

show_context = st.sidebar.checkbox(
    "Show Retrieved Context",
    value=True
)


# =====================================================
# EXAMPLES
# =====================================================

st.markdown("### 📌 Example Questions")

examples = [
    "Ai là vị vua đầu tiên của nhà Nguyễn?",
    "Lý Thái Tổ dời đô năm nào?",
    "Ai chỉ huy trận Như Nguyệt?",
    "Nhà Nguyễn kết thúc năm nào?",
    "Ai là người sáng lập nhà Lý?"
]

cols = st.columns(len(examples))

selected_question = None

for i, ex in enumerate(examples):

    if cols[i].button(ex):

        selected_question = ex


# =====================================================
# INPUT
# =====================================================

question = st.text_input(
    "🔍 Nhập câu hỏi lịch sử Việt Nam",
    value=selected_question if selected_question else ""
)


# =====================================================
# ASK
# =====================================================

if st.button("🚀 Ask"):

    if question.strip() == "":

        st.warning("Vui lòng nhập câu hỏi")

    else:

        with st.spinner("Đang truy xuất dữ liệu..."):

            retrieved = retrieve(
                question,
                top_k=top_k
            )

            context = "\n".join(retrieved)

        with st.spinner("Đang sinh câu trả lời..."):

            answer = generate_answer(
                question,
                context
            )

        st.markdown("---")

        st.markdown("## 📌 Answer")

        st.success(answer)

        if show_context:

            st.markdown("---")

            st.markdown("## 📚 Retrieved Context")

            for i, chunk in enumerate(retrieved):

                with st.expander(
                    f"Chunk {i+1}"
                ):

                    st.write(chunk)