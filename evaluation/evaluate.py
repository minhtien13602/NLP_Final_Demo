import json
import time


# =====================================================
# LOAD TEST SET
# =====================================================

with open(
    "evaluation/test_qa.json",
    "r",
    encoding="utf-8"
) as f:

    test_data = json.load(f)


# =====================================================
# IMPORT CONFIGS
# =====================================================

from experiments.A_base_no_rag import ask_A
from experiments.B_base_rag import ask_B
from experiments.C_ft_no_rag import ask_C
from experiments.D_ft_rag import ask_D


configs = {

    "A_Base_No_RAG": ask_A,

    "B_Base_RAG": ask_B,

    "C_FT_No_RAG": ask_C,

    "D_FT_RAG": ask_D
}


summary = []


# =====================================================
# EVALUATE
# =====================================================

for config_name, ask_fn in configs.items():

    print("=" * 50)

    print("Evaluating:", config_name)


    correct = 0

    total_time = 0


    for sample in test_data:

        question = sample["question"]

        gt = sample["answer"]


        # =============================================
        # INFERENCE
        # =============================================

        start = time.time()

        pred = ask_fn(question)

        end = time.time()


        infer_time = end - start

        total_time += infer_time


        # =============================================
        # EXACT MATCH
        # =============================================

        gt_clean = gt.lower().strip()

        pred_clean = pred.lower().strip()


        gt_words = set(
            gt_clean.split()
        )

        pred_words = set(
            pred_clean.split()
        )

        overlap = gt_words.intersection(
            pred_words
        )

        is_correct = (
            len(overlap) >= max(
                1,
                len(gt_words) // 2
            )
        )


        if is_correct:

            correct += 1


        print()

        print("Q:", question)

        print("GT:", gt)

        print("Pred:", pred)

        print("Correct:", is_correct)


    # =============================================
    # METRICS
    # =============================================

    accuracy = correct / len(test_data)

    avg_time = total_time / len(test_data)


    summary.append({

        "config": config_name,

        "accuracy": round(accuracy * 100, 2),

        "avg_time": round(avg_time, 2)
    })


# =====================================================
# SAVE
# =====================================================

with open(
    "evaluation/summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


print()

print("Saved evaluation/summary.json")