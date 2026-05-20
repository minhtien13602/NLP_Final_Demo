import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
# MODEL

MODEL_NAME = "Qwen/Qwen2-0.5B"

# LOAD TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token

# LOAD MODEL

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LORA CONFIG


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# LOAD DATASET

dataset = load_dataset(
    "json",
    data_files="data/train.json"
)

# FORMAT PROMPT

def format_prompt(example):

    prompt = f"""
### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}
"""

    return {"text": prompt}

dataset = dataset.map(format_prompt)

# TOKENIZE

def tokenize_function(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function)

# TRAINING ARGUMENTS

training_args = TrainingArguments(
    output_dir="./finetuned_model",

    per_device_train_batch_size=2,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    num_train_epochs=3,

    logging_steps=10,

    save_steps=50,

    fp16=True,

    report_to="none"
)

# DATA COLLATOR

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# TRAINER

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator
)

# TRAIN

trainer.train()

# SAVE MODEL

trainer.save_model("./finetuned_model")

tokenizer.save_pretrained("./finetuned_model")

print("Training completed!")