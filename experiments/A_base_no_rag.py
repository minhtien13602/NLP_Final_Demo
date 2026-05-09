import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)


# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("DEVICE:", DEVICE)


# =====================================================
# LOAD MODEL
# =====================================================

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
).to(DEVICE)


model.eval()

print("Base model loaded")


# =====================================================
# ASK FUNCTION
# =====================================================

def ask_A(question):

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

        answer = ask_A(question)

        print("\nAnswer:", answer)