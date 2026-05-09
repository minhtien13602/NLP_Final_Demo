import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)


# =====================================================
# PATH
# =====================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

FT_MODEL = ROOT_DIR / "finetuned_model"


# =====================================================
# LOAD MODEL
# =====================================================

BASE_MODEL = "Qwen/Qwen2-0.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32
).to(DEVICE)


model = PeftModel.from_pretrained(
    base_model,
    str(FT_MODEL)
)


model.eval()

print("Fine-tuned model loaded")


# =====================================================
# ASK FUNCTION
# =====================================================

def ask_C(question):

    prompt = f"""
Trả lời thật ngắn gọn.

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

        answer = ask_C(question)

        print("\nAnswer:", answer)