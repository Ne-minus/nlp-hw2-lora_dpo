from transformers import GPTNeoXForCausalLM, AutoTokenizer
from src.lora.utils import add_lora_to_model, mark_trainable, count_trainable_params, add_lora_to_neox
from src.dataset.dataset import PythiaSupervisedCollator
from src.sft.utils import train_epoch, evaluate
from torch.utils.data import DataLoader
from datasets import load_dataset

import torch
import wandb

wandb.init(
    project="pythia-lora-ft",
    name="pythia-1.4b-run-early_stopping-datafix",
)

model_name = "EleutherAI/pythia-1.4b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPTNeoXForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)
device = "cuda" if torch.cuda.is_available() else "cpu"

add_lora_to_model(model)
mark_trainable(model)
model.to(device)
print(model)
print("MODEL DTYPE:", next(model.parameters()).dtype)

train_dataset = load_dataset("json", data_files="src/dataset/files/hh_rlhf_chosen_train.jsonl")["train"]
test_dataset  = load_dataset("json", data_files="src/dataset/files/hh_rlhf_chosen_test.jsonl")["train"]

collator = PythiaSupervisedCollator(tokenizer)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)


train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collator
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collator
)

batch = next(iter(train_loader))

print("input_ids:", batch["input_ids"].dtype, batch["input_ids"].shape)
print("labels:", batch["labels"].dtype, batch["labels"].shape)

print("Sample input_ids:", batch["input_ids"][0][:50])
print("Sample labels:", batch["labels"][0][:50])

for epoch in range(1):
    train_loss, stopped, best_loss = train_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        val_dataloader=test_loader,
        eval_every=200,
        patience_steps=600,
        epoch=epoch,
        ckpt_path=f"best_step.pt"
    )

    if stopped:
        print("Early stopping: ending training early!")
        break