from datasets import load_dataset
import os
import json

def split_dialogue(lines):
    text = lines.strip("\n")

    last_pos = text.rfind("Assistant:")
    if last_pos == -1:
        return None

    target = text[last_pos + len("Assistant:"):].strip()
    input_text = text[:last_pos + len("Assistant:")].strip()


    return input_text, target


def process_split(split_name):
    path = "src/dataset/files/"
    
    os.makedirs(path, exist_ok=True)
    ds = load_dataset("Anthropic/hh-rlhf", split=split_name)

    chosen_out  = open(f"{path}hh_rlhf_chosen_{split_name}.jsonl", "w", encoding="utf8")
    rejected_out = open(f"{path}hh_rlhf_rejected_{split_name}.jsonl", "w", encoding="utf8")

    for entry in ds:
        # -------- chosen --------
        chosen = entry["chosen"]
        result = split_dialogue(chosen)

        if result is not None:
            input_text, target_text = result
            chosen_out.write(
                json.dumps({"input": input_text, "target": target_text}, ensure_ascii=False)
                + "\n"
            )

        # -------- rejected --------
        rejected = entry["rejected"]
        result_r = split_dialogue(rejected)

        if result_r is not None:
            input_text, target_text = result_r
            rejected_out.write(
                json.dumps({"input": input_text, "target": target_text}, ensure_ascii=False)
                + "\n"
            )

    chosen_out.close()
    rejected_out.close()

    print(f"{split_name}: готово")


class PythiaSupervisedCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        texts = []
        input_lengths = []

        for ex in examples:
            input_text = ex["input"].strip()
            target_text = ex["target"].strip()
            
            full_text = input_text + "\n" + target_text
            texts.append(full_text)

            input_tokens = self.tokenizer(
                input_text + "\n",
                add_special_tokens=False
            )["input_ids"]

            input_lengths.append(len(input_tokens))

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        labels = batch["input_ids"].clone()

        for i, inp_len in enumerate(input_lengths):
            labels[i, :inp_len] = -100

        batch["labels"] = labels
        return batch



if __name__ == "__main__":
    process_split("train")
    process_split("test")
