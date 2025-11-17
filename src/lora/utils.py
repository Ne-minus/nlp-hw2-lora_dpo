from typing import List
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from src.lora.lora_module import LoRALinear

def get_parents(model, module_name: str):

    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def add_lora_to_model(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
):
    replaced = []

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent, child_name = get_parents(model, name)
            old_layer = getattr(parent, child_name)
            lora_layer = LoRALinear(old_layer, r=r, alpha=alpha, dropout=dropout)
            setattr(parent, child_name, lora_layer)
            replaced.append(name)

    # print("LoRA добавлена в слои:")
    # for n in replaced:
    #     print("  -", n)

def add_lora_to_neox(
    model,
    r=8,
    alpha=16,
    dropout=0.0
):
    for name, module in model.named_modules():
        if "query_key_value" in name:
            parent, child_name = find_module(model, name)
            old = getattr(parent, child_name)
            new = QKV_LoRA(old, r, alpha, dropout)   # специальный LoRA
            setattr(parent, child_name, new)

        elif any(x in name for x in [
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h"
        ]):
            parent, child_name = find_module(model, name)
            old = getattr(parent, child_name)
            new = LoRALinear(old, r, alpha, dropout)
            setattr(parent, child_name, new)

def find_module(model, name):
    parts = name.split(".")
    for p in parts[:-1]:
        model = getattr(model, p)
    return model, parts[-1]


def mark_trainable(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for p in module.lora_A.parameters():
                p.requires_grad = True
            for p in module.lora_B.parameters():
                p.requires_grad = True


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, optimizer, train_loader, device, epochs=1, mode="experiment"):
    model.train()

    forwards = []
    backwards = []
    losses = []

    vram_forward = []
    vram_backward = []
    vram_step_total = []

    for epoch in range(epochs):
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            batch = {k: v.to(device) for k, v in batch.items()}

            torch.cuda.reset_peak_memory_stats()
            mem_base = get_mem()

            start_forward = time.time()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            end_forward = time.time()
            forwards.append(end_forward - start_forward)

            forward_peak = torch.cuda.max_memory_allocated() / 1024**2
            vram_forward.append(forward_peak)

            torch.cuda.reset_peak_memory_stats()
            start_backward = time.time()

            loss = outputs.loss
            loss.backward()

            end_backward = time.time()
            backwards.append(end_backward - start_backward)

            backward_peak = torch.cuda.max_memory_allocated() / 1024**2
            vram_backward.append(backward_peak)

            optimizer.step()
            optimizer.zero_grad()

            mem_end = get_mem()
            vram_step_total.append(mem_end - mem_base)

            if step % 50 == 0:
                losses.append(loss.detach().cpu().numpy())

            if mode == "experiment" and step == 1000:
                break

    return {
        "forwards": forwards,
        "backwards": backwards,
        "losses": losses,
        "vram_forward": vram_forward,
        "vram_backward": vram_backward,
        "vram_step_total": vram_step_total,
    }



def get_mem():
    return torch.cuda.memory_allocated() / 1024**2 

            
