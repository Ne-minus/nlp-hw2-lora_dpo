from typing import List
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from lora.lora_module import LoRALinear

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
    for epoch in range(epochs):
        losses = []
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            batch = {k: v.to(device) for k, v in batch.items()}
            start_forward = time.time()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            end_forward = time.time()
            forward_duration = end_forward - start_forward
            forwards.append(forward_duration)

            start_backward = time.time()
            loss = outputs.loss

            loss.backward()
            end_backward = time.time()
            backward_duration = end_backward - start_backward
            backwards.append(backward_duration)

            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                losses.append(loss.detach().cpu().numpy())

            if mode == "experiment" and step == 1000:
                break

    return forwards, backwards, losses

            
