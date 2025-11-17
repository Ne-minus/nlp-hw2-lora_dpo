from src.lora.utils import get_parents, add_lora_to_model, mark_trainable, count_trainable_params, train, get_mem
from src.lora.lora_module import LoRALinear
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
import torch
import copy
import numpy as np


def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

if __name__ == "__main__":
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")


    model_name = "Qwen/Qwen1.5-0.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=8,
        shuffle=True,
        collate_fn=collator
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)


    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    model_lora = copy.deepcopy(model)

    model.to(device)

    add_lora_to_model(model_lora)
    mark_trainable(model_lora)
    model_lora.to(device)

    print("Число параметров для базовой модели: ", count_trainable_params(model))
    print("Число параметров для LoRA модели: ", count_trainable_params(model_lora))


    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer_lora = torch.optim.AdamW(model_lora.parameters(), lr=5e-5)


    data = train(model, optimizer, train_loader, device)
    data_lora = train (model_lora, optimizer_lora, train_loader, device)

    avg_forward = np.mean(data["forwards"])
    avg_backward = np.mean(data["backwards"])
    avg_losses = np.mean(data["losses"])
    

    avg_forward_lora = np.mean(data_lora["forwards"])
    avg_backward_lora = np.mean(data_lora["backwards"])
    avg_losses_lora = np.mean(data_lora["losses"])

    avg_vram_forward = np.mean(data["vram_forward"])
    avg_vram_backward = np.mean(data["vram_backward"])
    avg_vram_step_total = np.mean(data["vram_step_total"])

    avg_vram_forward_lora = np.mean(data_lora["vram_forward"])
    avg_vram_backward_lora = np.mean(data_lora["vram_backward"])
    avg_vram_step_total_lora = np.mean(data_lora["vram_step_total"])


    for i, loss in enumerate([data["losses"], data_lora["losses"]]): 
        plt.figure(figsize=(8, 5))
        plt.plot(loss, label="Loss", linewidth=1.5)
        plt.xlabel("Итерация")
        plt.ylabel("Значение Loss")
        if i == 0:
            plt.title("Убывание ошибки")
        else:
            plt.title("Убывание ошибки (LoRA fine-tuning)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # при необходимости можно сохранить:
        plt.savefig(f"{i}_loss_curve.png", dpi=150)

    print("=== Базовая модель ===")
    print(f"Среднее время forward:      {avg_forward:.4f}")
    print(f"Среднее время backward:     {avg_backward:.4f}")
    print(f"Средний лосс:               {avg_losses:.4f}")
    # print(f"Средний VRAM forward (MB):  {avg_vram_forward:.2f}")
    # print(f"Средний VRAM backward (MB): {avg_vram_backward:.2f}")
    print(f"Средний VRAM step (MB):     {avg_vram_step_total:.2f}")

    print("\n=== Модель с LoRA ===")
    print(f"Среднее время forward:      {avg_forward_lora:.4f}")
    print(f"Среднее время backward:     {avg_backward_lora:.4f}")
    print(f"Средний лосс:               {avg_losses_lora:.4f}")
    # print(f"Средний VRAM forward (MB):  {avg_vram_forward_lora:.2f}")
    # print(f"Средний VRAM backward (MB): {avg_vram_backward_lora:.2f}")
    print(f"Средний VRAM step (MB):     {avg_vram_step_total_lora:.2f}")




