import torch
from torch.nn.utils import clip_grad_norm_

import wandb
from tqdm import tqdm

def train_epoch(
    model,
    dataloader,
    optimizer,
    device,
    val_dataloader=None,      # добавили для early stopping
    eval_every=200,           # каждые 200 шагов — валидация
    patience_steps=600,       # если 600 шагов без улучшения — стоп
    max_grad_norm=1.0,
    epoch=0,
    ckpt_path="best_step_ckpt_new.pt"
):
    model.train()

    total_loss = 0.0
    steps = 0
    global_step = epoch * len(dataloader)

    # Early stopping state
    best_val_loss = float("inf")
    steps_without_improve = 0
    early_stopped = False

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):

        global_step += 1

        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss

        loss.backward()

        clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # wandb
        if wandb.run and step % 50 == 0:
            wandb.log({
                "train/loss_step": float(loss.detach().cpu()),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step": global_step,
                "gpu_mem": torch.cuda.memory_allocated() / 1e9,
            })

        total_loss += loss.item()
        steps += 1

        # ---- EARLY STOPPING EVERY eval_every STEPS ----
        if (
            val_dataloader is not None and
            global_step % eval_every == 0
        ):
            val_loss = evaluate_fast(model, val_dataloader, device)

            if wandb.run:
                wandb.log({"val/loss": val_loss, "step": global_step})

            print(f"[step {global_step}] val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                steps_without_improve = 0

                print(f"  ↳ New best val loss! Saving checkpoint → {ckpt_path}")
                torch.save(model.state_dict(), ckpt_path)

                if wandb.run:
                    wandb.log({"early_stopping/best_val_loss": best_val_loss})
            else:
                steps_without_improve += eval_every
                print(f"  ↳ No improvement for {steps_without_improve} steps")

                if steps_without_improve >= patience_steps:
                    print("\n🔥 Early stopping triggered (inside epoch)!")
                    early_stopped = True
                    break

    avg_loss = total_loss / steps

    if wandb.run:
        wandb.log({
            "train/loss_epoch": avg_loss,
            "train/epoch": epoch
        })

    return avg_loss, early_stopped, best_val_loss



@torch.no_grad()
def evaluate(model, dataloader, device, epoch=0):
    model.eval()
    total_loss = 0
    steps = 0

    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss
        total_loss += loss.item()
        steps += 1

    avg_loss = total_loss / steps

    wandb.log({
        "eval/loss": avg_loss,
        "eval/epoch": epoch
    })

    model.train()
    return avg_loss


def evaluate_fast(model, dataloader, device, max_batches=50):
    model.eval()
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )

            total_loss += outputs.loss.item()
            steps += 1

    return total_loss / steps