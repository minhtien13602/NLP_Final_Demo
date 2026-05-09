# 🇻🇳 Vietnam History QA RAG

Hệ thống hỏi đáp lịch sử Việt Nam sử dụng:

- RAG (Retrieval-Augmented Generation)
- FAISS Vector Search
- SBERT Embedding
- Fine-tuning Qwen2-0.5B bằng LoRA
- Streamlit UI


# 📌 Features

- Semantic Search bằng SBERT
- Vector Database bằng FAISS
- RAG Question Answering
- Fine-tuned Vietnamese QA
- Streamlit demo UI
- Benchmark 4 cấu hình:
  - Base Model
  - Base + RAG
  - Fine-tuned
  - Fine-tuned + RAG


# 📂 Project Structure

vietnam_qa_rag/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── chunks.json
│       └── faiss.index
│
├── experiments/
│   ├── A_base_no_rag.py
│   ├── B_base_rag.py
│   ├── C_ft_no_rag.py
│   └── D_ft_rag.py
│
├── evaluation/
│   ├── evaluate.py
│   └── test_qa.json
│
├── finetuned_model/
│
├── rag/
│   └── build_db.py
│
├── scripts/
│   ├── crawl_wiki.py
│   ├── clean_text.py
│   └── build_faiss.py
│
├── app/
│   └── app.py
│
├── requirements.txt
│
└── README.md


# ⚙️ Installation

## 1. Clone repository

git clone https://github.com/minhtien13602/NLP_Final_Demo.git

cd NLP_Final_Demo

## 2. Create virtual environment

### Windows

python -m venv venv

Activate:

venv\\Scripts\\activate


## 3. Install dependencies

pip install -r requirements.txt

# 📚 Build RAG Database

## 1. Prepare data

📚 Data Crawling

Project hỗ trợ crawl dữ liệu lịch sử trực tiếp từ Wikipedia.

Run crawler:

python scripts/crawl_wiki.py

Kết quả:

data/raw/nha_ly.txt
data/raw/nha_nguyen.txt

🧹 Clean Raw Data

Sau khi crawl dữ liệu, thực hiện làm sạch văn bản.

Run cleaning script:

python scripts/clean_text.py

Kết quả:

data/raw/history_clean.txt

## 2. Build chunks

python rag/build_db.py

Kết quả:

data/processed/chunks.json

## 3. Build FAISS index

python scripts/build_faiss.py

Kết quả:

data/processed/faiss.index

# 🤖 Fine-tuning

Fine-tuning được thực hiện bằng:

- LoRA
- PEFT
- Qwen2-0.5B-Instruct

Model sau train được lưu tại:

finetuned_model/


# 🧪 Evaluation

## Benchmark 4 configurations

| Config |   Description    |
|--------|------------------|
|    A   | Base Model       |
|    B   | Base + RAG       |
|    C   | Fine-tuned       |
|    D   | Fine-tuned + RAG |


## Run evaluation

python -m evaluation.evaluate

Kết quả:

evaluation/summary.json


# 🚀 Run Streamlit App

streamlit run app/app.py

Mở trình duyệt:

http://localhost:8501


# 🧠 Technologies Used

- Python
- Transformers
- PEFT
- LoRA
- SentenceTransformers
- FAISS
- Streamlit
- HuggingFace


# 📌 Observations

- RAG giúp giảm hallucination đáng kể.
- Fine-tuning giúp mô hình hiểu domain lịch sử tốt hơn.
- Retrieval quality ảnh hưởng mạnh tới factual QA.
- Tiny LLM vẫn gặp hạn chế với descriptive QA.


# 🔮 Future Improvements

- Hybrid Search (BM25 + Dense Retrieval)
- Cross Encoder Reranker
- PhoBERT QA
- Larger LLM
- More training data
- Deploy online


# 👨‍💻 Author

Nguyễn Minh Tiến


# 📄 License

MIT License