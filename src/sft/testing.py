import torch
import json
import random
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from src.lora.utils import add_lora_to_model, mark_trainable, count_trainable_params, add_lora_to_neox



# ======================
# CONFIG
# ======================
MODEL_NAME = "EleutherAI/pythia-1.4b"     # или твой путь
CKPT_PATH = "/workspace/best_step.pt"                        # чекпоинт после обучения
TEST_PATH = "/workspace/src/dataset/files/hh_rlhf_chosen_test.jsonl"                          # твой test датасет
NUM_EXAMPLES = 5                                   # сколько примеров показать
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================
# LOAD MODEL + TOKENIZER
# ======================
print("Loading tokenizer & base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = GPTNeoXForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

add_lora_to_model(model)
model.to(DEVICE)
model.eval()

print("Loading checkpoint:", CKPT_PATH)
state = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(state)
print("Checkpoint loaded successfully.")


# ======================
# LOAD TEST DATASET
# ======================
print("Loading test dataset:", TEST_PATH)
test_samples = []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        test_samples.append(json.loads(line))

print(f"Loaded {len(test_samples)} test samples.")


# ======================
# GENERATION FUNCTION
# ======================
def generate_answer(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt + "\n Assistant: ", return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=None,
        )
        print("OUTPUT: ", output)

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ======================
# RUN RANDOM EXAMPLES
# ======================
examples = random.sample(test_samples, NUM_EXAMPLES)

for idx, ex in enumerate(examples):
    print("\n" + "=" * 50)
    print(f"EXAMPLE {idx+1}")
    print("=" * 50)

    print("\nINPUT:")
    print(ex["input"])

    print("\nTARGET:")
    print(ex["target"])

    print("\nMODEL OUTPUT:")
    out = generate_answer(model, tokenizer, ex["input"])
    print(out)

print("\nDone.")
