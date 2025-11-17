from huggingface_hub import upload_file

repo_id = "Kate-03/pythia-sft-lora"   # замените на свой

upload_file(
    path_or_fileobj="/workspace/best_step.pt",   # локальный файл
    path_in_repo="/workspace/best_step.pt",      # имя в репозитории
    repo_id=repo_id,
    repo_type="model"
)

print("Готово!")
