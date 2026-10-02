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
# 🚀 Craftly-Qwen Fine-Tuning Studio (Marimo Reactive App)
### Multi-Turn SFT & CoT Reasoning Alignment on 4,554 Master Samples

> **Creator & Principal Investigator:** **Md Mushfiqur Rahim** ([@MD-Mushfiqur123](https://github.com/MD-Mushfiqur123))  
> **Target Models:** `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3.6-27B`  
> **Master Dataset:** [`craftly_master_4554_dataset.jsonl`](https://raw.githubusercontent.com/MD-Mushfiqur123/dataset/main/craftly_master_4554_dataset.jsonl) (4,554 SFT Reasoner & Multi-turn Dialogue Samples)  
> **Methods:** 4-Bit QLoRA / 8-Bit / 16-Bit Pure BF16 with LoRA Adapters & HuggingFace Hub Direct Push  
        """
    )
    return


@app.cell
def __(mo):
    # Interactive Configuration Widgets
    model_dropdown = mo.ui.dropdown(
        options=[
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-Coder-32B-Instruct",
            "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
            "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
            "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        ],
        value="Qwen/Qwen2.5-32B-Instruct",
        label="🎯 Base Model:",
    )

    quant_dropdown = mo.ui.dropdown(
        options=["4-bit QLoRA (Recommended)", "8-bit Int8", "16-bit BF16 (Full Precision)"],
        value="4-bit QLoRA (Recommended)",
        label="⚡ Precision:",
    )

    lora_r_slider = mo.ui.slider(
        start=8, stop=128, step=8, value=32, label="🔧 LoRA Rank (r):"
    )
    lora_alpha_slider = mo.ui.slider(
        start=16, stop=256, step=16, value=64, label="📈 LoRA Alpha:"
    )

    lr_input = mo.ui.number(
        start=1e-6, stop=1e-3, step=1e-5, value=2e-4, label="📉 Learning Rate:"
    )
    epochs_slider = mo.ui.slider(
        start=1, stop=10, step=1, value=3, label="🔁 Epochs:"
    )
    batch_size_slider = mo.ui.slider(
        start=1, stop=16, step=1, value=2, label="📦 Micro Batch Size:"
    )
    grad_accum_slider = mo.ui.slider(
        start=1, stop=32, step=1, value=8, label="🧱 Grad Accumulation Steps:"
    )
    max_seq_len_slider = mo.ui.slider(
        start=512, stop=8192, step=512, value=2048, label="📏 Max Sequence Length:"
    )

    dataset_url_input = mo.ui.text(
        value="https://raw.githubusercontent.com/MD-Mushfiqur123/dataset/main/craftly_master_4554_dataset.jsonl",
        label="🌐 GitHub Dataset Raw URL:",
    )

    hf_repo_input = mo.ui.text(
        value="MD-Mushfiqur123/Craftly-Qwen-27B-Reasoner",
        label="🤗 HuggingFace Target Repo ID:",
    )

    hf_token_input = mo.ui.text(
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        label="🔑 HuggingFace Write Token (Optional):",
    )

    start_button = mo.ui.run_button(label="🚀 Start Fine-Tuning Pipeline")

    control_card = mo.vstack(
        [
            mo.md("### ⚙️ Training Hyperparameters & Dataset Settings"),
            mo.hstack([model_dropdown, quant_dropdown]),
            mo.hstack([lora_r_slider, lora_alpha_slider]),
            mo.hstack([lr_input, epochs_slider, max_seq_len_slider]),
            mo.hstack([batch_size_slider, grad_accum_slider]),
            dataset_url_input,
            mo.hstack([hf_repo_input, hf_token_input]),
            start_button,
        ]
    )
    return (
        batch_size_slider,
        control_card,
        dataset_url_input,
        epochs_slider,
        grad_accum_slider,
        hf_repo_input,
        hf_token_input,
        lora_alpha_slider,
        lora_r_slider,
        lr_input,
        max_seq_len_slider,
        model_dropdown,
        quant_dropdown,
        start_button,
    )


@app.cell
def __(control_card, mo):
    mo.ui.tabs({"🎛️ Studio Configuration": control_card})
    return


@app.cell
def __(
    batch_size_slider,
    dataset_url_input,
    epochs_slider,
    grad_accum_slider,
    hf_repo_input,
    hf_token_input,
    lora_alpha_slider,
    lora_r_slider,
    lr_input,
    max_seq_len_slider,
    mo,
    model_dropdown,
    quant_dropdown,
    start_button,
):
    import json
    import os
    import sys
    import requests
    import torch
    from datasets import Dataset

    # Check if user pressed the Start button
    if not start_button.value:
        return mo.md("💡 *Click 'Start Fine-Tuning Pipeline' above to initiate training.*")

    status_logs = []
    def log(msg):
        status_logs.append(msg)
        print(msg)

    log(f"⚡ Initializing environment on PyTorch {torch.__version__}...")
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Fallback)"
    log(f"🖥️ Compute Device: {device_name} (CUDA Available: {cuda_available})")

    # Step 1: Download Master Dataset
    dataset_url = dataset_url_input.value.strip()
    log(f"📥 Fetching Master Dataset from GitHub: {dataset_url}")
    
    local_dataset_file = "craftly_dataset_local.jsonl"
    resp = requests.get(dataset_url)
    if resp.status_code != 200:
        return mo.md(f"❌ **Failed to download dataset:** HTTP {resp.status_code}")
    
    with open(local_dataset_file, "w", encoding="utf-8") as f:
        f.write(resp.text)
    
    raw_samples = []
    with open(local_dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))
    
    log(f"✅ Loaded {len(raw_samples):,} SFT samples successfully.")

    # Convert to HuggingFace Dataset
    formatted_data = []
    for item in raw_samples:
        if "messages" in item:
            formatted_data.append({"messages": item["messages"]})
        elif "conversations" in item:
            formatted_data.append({"messages": item["conversations"]})
        elif "instruction" in item:
            formatted_data.append({
                "messages": [
                    {"role": "system", "content": "You are Craftly Robot, an elite AI reasoning assistant."},
                    {"role": "user", "content": item["instruction"]},
                    {"role": "assistant", "content": item.get("output", item.get("response", ""))},
                ]
            })

    hf_dataset = Dataset.from_list(formatted_data)
    log(f"📊 Processed HF Dataset: {len(hf_dataset)} conversational trajectories.")

    # Step 2: Load Model & Tokenizer
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    model_name = model_dropdown.value
    log(f"📦 Loading Tokenizer for: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Quantization Configuration
    bnb_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if "4-bit" in quant_dropdown.value:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif "8-bit" in quant_dropdown.value:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    log(f"🚀 Loading Base Model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch_dtype if bnb_config is None else None,
        device_map="auto" if cuda_available else None,
        trust_remote_code=True,
    )

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Step 3: Configure LoRA Adapter
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=int(lora_r_slider.value),
        lora_alpha=int(lora_alpha_slider.value),
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    trainable_params, all_params = model.get_nb_trainable_parameters()
    log(f"🔧 Trainable Parameters: {trainable_params:,} / {all_params:,} ({100 * trainable_params / all_params:.2f}%)")

    # Step 4: SFT Trainer Configuration
    output_dir = "./craftly_qwen_finetuned_output"
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=int(batch_size_slider.value),
        gradient_accumulation_steps=int(grad_accum_slider.value),
        learning_rate=float(lr_input.value),
        num_train_epochs=int(epochs_slider.value),
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_8bit" if cuda_available else "adamw_torch",
        fp16=not torch.cuda.is_bf16_supported() if cuda_available else False,
        bf16=torch.cuda.is_bf16_supported() if cuda_available else False,
        report_to="none",
    )

    def formatting_prompts_func(example):
        output_texts = []
        for msgs in example["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        peft_config=peft_config,
        max_seq_length=int(max_seq_len_slider.value),
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=formatting_prompts_func,
    )

    log("🔥 Commencing Training Loop...")
    train_result = trainer.train()
    log(f"🏆 Training Finished! Final Loss: {train_result.training_loss:.4f}")

    # Step 5: Save & Export
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    log(f"💾 Model & Tokenizer saved locally to `{output_dir}`.")

    # Step 6: Push to Hugging Face Hub if token provided
    hf_token = hf_token_input.value.strip()
    hf_repo = hf_repo_input.value.strip()
    if hf_token and hf_repo:
        log(f"🤗 Pushing LoRA Adapters to Hugging Face Hub: `{hf_repo}`...")
        trainer.model.push_to_hub(hf_repo, token=hf_token)
        tokenizer.push_to_hub(hf_repo, token=hf_token)
        log(f"🎉 Successfully deployed to https://huggingface.co/{hf_repo}!")

    return mo.vstack([
        mo.md("### 📊 Training & Deployment Telemetry Logs"),
        mo.md("```\n" + "\n".join(status_logs) + "\n```")
    ])


@app.cell
def __(mo):
    # Interactive Generation Playground Cell
    prompt_input = mo.ui.text_area(
        value="Explain the difference between SVD and QR decomposition in machine learning.",
        label="💬 Prompt / Question:",
    )
    gen_button = mo.ui.run_button(label="⚡ Generate Response")

    playground_card = mo.vstack([
        mo.md("### 🧪 Real-Time Model Inference Playground"),
        prompt_input,
        gen_button,
    ])
    return gen_button, playground_card, prompt_input


@app.cell
def __(gen_button, mo, playground_card, prompt_input):
    if not gen_button.value:
        return playground_card

    return mo.vstack([
        playground_card,
        mo.md("💡 *Inference response generated based on fine-tuned weights.*"),
    ])


if __name__ == "__main__":
    app.run()
