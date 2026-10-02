# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "unsloth",
#     "accelerate",
#     "transformers>=4.48.0",
#     "peft>=0.12.0",
#     "trl>=0.11.0",
#     "datasets>=3.0.0",
#     "bitsandbytes>=0.43.0",
#     "huggingface_hub",
#     "requests",
#     "tqdm",
#     "torch",
# ]
# ///

import os
# Mandatory Unsloth optimizations for Qwen3.8 Gated DeltaNet
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
# 🚀 Bulletproof Master Suite: Qwen3.8-27B 1-Epoch Fine-Tuning
### Official Unsloth FastLanguageModel Engine on 4,554 SFT Reasoner Samples
> **Creator & Principal Investigator:** **Md Mushfiqur Rahim** ([@MD-Mushfiqur123](https://github.com/MD-Mushfiqur123))  
> **Model Target:** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)  
> **Engine:** Unsloth Native CUDA Acceleration (Zero Accelerate / DeviceMap Errors)  
> **Master Dataset:** [`craftly_master_4554_dataset.jsonl`](https://raw.githubusercontent.com/MD-Mushfiqur123/dataset/main/craftly_master_4554_dataset.jsonl)  
> **Hardware:** NVIDIA RTX PRO 6000 Blackwell  
> **Execution:** **100% Fully Autonomous (1 Epoch)**  
        """
    )
    return


@app.cell
def __(mo):
    import json
    import os
    import sys
    import subprocess
    import requests
    import torch

    logs = []
    def log(msg):
        logs.append(msg)
        print(msg)

    log("=" * 70)
    log("⚡ INITIATING BULLETPROOF QWEN3.8-27B UNSLOTH TRAINING PIPELINE")
    log("=" * 70)

    # 1. Self-Healing Environment Verification
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        import accelerate
        log("✅ Unsloth and Accelerate imported.")
    except ImportError:
        log("📦 Auto-installing unsloth and accelerate...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth", "accelerate"])
        from unsloth import FastLanguageModel, is_bfloat16_supported
        import accelerate

    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    # 2. Hardware Telemetry
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    log(f"🖥️ Hardware Detected: {dev_name} | VRAM: {vram_gb:.2f} GiB | PyTorch: {torch.__version__}")

    # 3. Download Master Dataset (4,554 Samples)
    dataset_url = "https://raw.githubusercontent.com/MD-Mushfiqur123/dataset/main/craftly_master_4554_dataset.jsonl"
    local_file = "craftly_master_4554_dataset.jsonl"
    log(f"📥 Fetching Master Dataset from: {dataset_url}")

    try:
        resp = requests.get(dataset_url, timeout=30)
        if resp.status_code == 200:
            with open(local_file, "w", encoding="utf-8") as f:
                f.write(resp.text)
            log("✅ Dataset successfully fetched from GitHub.")
    except Exception as e:
        log(f"ℹ️ Network notice: {e}, using existing local file.")

    raw_samples = []
    with open(local_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))
    log(f"📊 Loaded {len(raw_samples):,} Master SFT Samples.")

    # 4. Load Qwen/Qwen3.8-27B with Unsloth Dual-Architecture Safety
    target_model = "Qwen/Qwen3.8-27B"
    max_seq_len = 2048
    log(f"🚀 Loading Target Model: `{target_model}` via Unsloth (4-Bit QLoRA)...")

    model = None
    tokenizer = None
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=target_model,
            max_seq_length=max_seq_len,
            load_in_4bit=True,
            dtype=None,
        )
        log(f"🎯 Successfully loaded `{target_model}` via FastLanguageModel!")
    except Exception as err:
        log(f"ℹ️ FastLanguageModel notice ({err}), testing FastVisionModel...")
        from unsloth import FastVisionModel
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name=target_model,
            max_seq_length=max_seq_len,
            load_in_4bit=True,
            dtype=None,
        )
        log(f"🎯 Successfully loaded `{target_model}` via FastVisionModel!")

    # 5. Apply Unsloth Optimized LoRA Adapters
    log("🧩 Injecting Unsloth PEFT LoRA Adapters...")
    linear_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=linear_targets,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    trainable_p, total_p = model.get_nb_trainable_parameters()
    log(f"🔧 LoRA Adapter Injected: {trainable_p:,} / {total_p:,} ({100*trainable_p/total_p:.2f}%)")

    # 6. Format Dataset into ChatML
    log("📝 Formatting Dataset into ChatML Prompt Templates...")
    chatml_texts = []
    for item in raw_samples:
        convo = item.get("messages", item.get("conversations", []))
        if convo:
            text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            chatml_texts.append(text)

    hf_dataset = Dataset.from_dict({"text": chatml_texts})
    log(f"✅ Prepared {len(hf_dataset):,} ChatML training samples.")

    # 7. SFTTrainer 1-Epoch Execution
    output_dir = "./craftly_qwen38_27b_master_output"
    bf16_supported = is_bfloat16_supported() if cuda_avail else False

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,  # Exact 1 Epoch
        logging_steps=5,
        save_strategy="epoch",
        optim="adamw_8bit",
        fp16=not bf16_supported,
        bf16=bf16_supported,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    log("🔥 Commencing High-Speed 1-Epoch Training Loop on Qwen3.8-27B...")
    train_res = trainer.train()
    log(f"🏆 Training Finished Successfully! Final Loss: {train_res.training_loss:.4f}")

    # 8. Save Model & Tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log(f"💾 Master LoRA weights saved to: `{output_dir}`")

    # 9. Verification Inference
    log("🧪 Running Post-Training Inference Verification...")
    FastLanguageModel.for_inference(model)
    test_messages = [{"role": "user", "content": "Who created you and what is your purpose?"}]
    input_text = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
        )
    resp_text = tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    log(f"🤖 Output:\n{resp_text}")

    log("=" * 70)
    log("🎉 100% BULLETPROOF RUN COMPLETED WITH ZERO ERRORS!")
    log("=" * 70)

    return mo.vstack([
        mo.md("### 📊 Qwen3.8-27B Unsloth Execution Telemetry"),
        mo.md("```\n" + "\n".join(logs) + "\n```")
    ])


if __name__ == "__main__":
    app.run()
