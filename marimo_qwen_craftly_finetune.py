# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "transformers>=4.48.0",
#     "accelerate>=0.34.0",
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
# 🚀 Final Master Autonomous Craftly-Qwen Fine-Tuning Suite
### 1-Epoch Master SFT & CoT Reasoning Alignment on Qwen-27B/32B
> **Creator & Principal Investigator:** **Md Mushfiqur Rahim** ([@MD-Mushfiqur123](https://github.com/MD-Mushfiqur123))  
> **Target Models:** `Qwen/Qwen3.8-27B` (Auto-fallback to `Qwen/Qwen2.5-32B-Instruct` / `14B` / `7B`)  
> **Master Dataset:** [`craftly_master_4554_dataset.jsonl`](https://raw.githubusercontent.com/MD-Mushfiqur123/dataset/main/craftly_master_4554_dataset.jsonl) (4,554 Samples)  
> **Execution Mode:** **Zero-Click 100% Autonomous Pipeline (1 Epoch)**  
        """
    )
    return


@app.cell
def __(mo):
    import json
    import os
    import sys
    import requests
    import torch
    from datasets import Dataset

    logs = []
    def log(msg):
        logs.append(msg)
        print(msg)

    log("=" * 70)
    log("🔥 LAUNCHING AUTONOMOUS CRAFTLY-QWEN 1-EPOCH MASTER SUITE")
    log("=" * 70)

    # 1. Hardware & Compute Device Telemetry
    cuda_avail = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if cuda_avail else 0.0
    log(f"🖥️ Hardware Detected: {dev_name} | VRAM: {vram_gb:.2f} GiB | PyTorch: {torch.__version__}")

    # 2. Download Master Dataset (4,554 SFT Reasoner Samples)
    dataset_url = "https://raw.githubusercontent.com/MD-Mushfiqur123/dataset/main/craftly_master_4554_dataset.jsonl"
    log(f"📥 Downloading Master Dataset from: {dataset_url}")
    
    local_file = "craftly_master_4554_dataset.jsonl"
    try:
        resp = requests.get(dataset_url, timeout=30)
        if resp.status_code == 200:
            with open(local_file, "w", encoding="utf-8") as f:
                f.write(resp.text)
            log("✅ Dataset successfully downloaded from GitHub.")
    except Exception as e:
        log(f"⚠️ Network check: {e}, falling back to local file...")

    raw_samples = []
    with open(local_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))
    log(f"📊 Total Training Samples Loaded: {len(raw_samples):,}")

    formatted_data = []
    for item in raw_samples:
        if "messages" in item:
            formatted_data.append({"messages": item["messages"]})
        elif "conversations" in item:
            formatted_data.append({"messages": item["conversations"]})
        elif "instruction" in item:
            formatted_data.append({
                "messages": [
                    {"role": "system", "content": "You are Craftly Robot, an elite AI reasoning assistant created by Md Mushfiqur Rahim."},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item.get("output", item.get("response", ""))},
                ]
            })

    hf_dataset = Dataset.from_list(formatted_data)
    log(f"✅ Formatted {len(hf_dataset):,} conversational trajectories.")

    # 3. Model & Tokenizer Resolution with Graceful Fallback
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    target_models = [
        "Qwen/Qwen3.8-27B",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]

    model_name = None
    tokenizer = None
    for candidate in target_models:
        try:
            log(f"🔍 Testing candidate: `{candidate}`...")
            tokenizer = AutoTokenizer.from_pretrained(candidate, trust_remote_code=True)
            model_name = candidate
            log(f"🎯 Successfully resolved target model: `{model_name}`")
            break
        except Exception as e:
            log(f"⏩ Candidate `{candidate}` unavailable or gated ({e}), attempting next candidate...")

    if model_name is None:
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. Quantization Configuration (4-Bit NF4)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    ) if cuda_avail else None

    log(f"🚀 Loading `{model_name}` onto {dev_name} (4-Bit NF4 QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if cuda_avail else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # 5. Robust LoRA Adapter (Targeting standard Linear Projection layers)
    linear_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=linear_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable_p, total_p = model.get_nb_trainable_parameters()
    log(f"🔧 LoRA Adapter Injected: {trainable_p:,} / {total_p:,} ({100*trainable_p/total_p:.2f}%)")

    # 6. SFTTrainer Self-Adaptive Arguments Setup
    output_dir = "./craftly_qwen_autonomous_output"
    
    def formatting_func(example):
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in example["messages"]
        ]

    trainer = None
    try:
        from trl import SFTConfig
        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=5,
            save_strategy="epoch",
            optim="paged_adamw_8bit" if cuda_avail else "adamw_torch",
            fp16=not torch.cuda.is_bf16_supported() if cuda_avail else False,
            bf16=torch.cuda.is_bf16_supported() if cuda_avail else False,
            max_seq_length=2048,
            dataset_text_field=None,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=hf_dataset,
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=sft_config,
            formatting_func=formatting_func,
        )
        log("⚙️ Initialized via modern SFTConfig pipeline.")
    except Exception as e:
        log(f"ℹ️ Fallback to standard TrainingArguments: {e}")
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=5,
            save_strategy="epoch",
            optim="paged_adamw_8bit" if cuda_avail else "adamw_torch",
            fp16=not torch.cuda.is_bf16_supported() if cuda_avail else False,
            bf16=torch.cuda.is_bf16_supported() if cuda_avail else False,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=hf_dataset,
            peft_config=peft_config,
            max_seq_length=2048,
            tokenizer=tokenizer,
            args=training_args,
            formatting_func=formatting_func,
        )

    # 7. Execute 1-Epoch Training Loop
    log("⚡ Commencing SFT Training Loop (1 Epoch)...")
    train_res = trainer.train()
    log(f"🏆 Training Complete! Final Loss: {train_res.training_loss:.4f}")

    # 8. Save Weights & Verification Inference
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log(f"💾 Master LoRA weights saved to: `{output_dir}`")

    log("🧪 Running Post-Training Inference Verification...")
    test_prompt = [{"role": "user", "content": "Who created you and what is your purpose?"}]
    input_text = tokenizer.apply_chat_template(test_prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    response_text = tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    log(f"🤖 Output:\n{response_text}")

    log("=" * 70)
    log("🎉 100% MASTER BUILD COMPLETE — READY FOR PRODUCTION DEPLOYMENT")
    log("=" * 70)

    return mo.vstack([
        mo.md("### 📊 Final Master Build Telemetry"),
        mo.md("```\n" + "\n".join(logs) + "\n```")
    ])


if __name__ == "__main__":
    app.run()
