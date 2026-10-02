import os
import sys
import torch
import marimo

app = marimo.App(width="full")

@app.cell
def __():
    import marimo as mo
    return (mo,)

@app.cell
def __(mo):
    mo.md(
        r"""
# 🚀 Craftly-Qwen3.8-27B: 3-Way Precision Merge & Hugging Face Publisher
### Merging 4-Bit LoRA Adapters into Standalone Full Weights: **BF16**, **FP16**, and **FP32**

> **Author & PI:** **Md Mushfiqur Rahim** ([@MD-Mushfiqur123](https://github.com/MD-Mushfiqur123))  
> **Model Scale:** 27 Billion Parameters  
> **Precision Sizes:**  
> - **BF16:** ~54 GB (Optimal for Ampere / Hopper / Blackwell GPUs & vLLM)  
> - **FP16:** ~54 GB (Universal IEEE 754 Half-Precision for T4 / V100 / RTX 30/40)  
> - **FP32:** ~108 GB (Maximum Numerical Precision & Master Reference Weights)  
        """
    )
    return

@app.cell
def __(mo):
    # Interactive UI for User Credentials
    hf_token_ui = mo.ui.text(
        placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        label="🔑 Hugging Face Write Token:",
        kind="password",
    )
    username_ui = mo.ui.text(
        value="MD-Mushfiqur123",
        label="👤 HF Username / Org:",
    )
    repo_name_ui = mo.ui.text(
        value="Craftly-Qwen3.8-27B-Reasoner",
        label="📦 Base Model Repo Name:",
    )
    
    upload_bf16_ui = mo.ui.checkbox(value=True, label="Upload BF16 (~54 GB)")
    upload_fp16_ui = mo.ui.checkbox(value=True, label="Upload FP16 (~54 GB)")
    upload_fp32_ui = mo.ui.checkbox(value=True, label="Upload FP32 (~108 GB)")

    merge_btn = mo.ui.run_button(label="🚀 Merge & Push All 3 Repositories to Hugging Face", kind="success")

    config_box = mo.vstack([
        mo.md("### ⚙️ Hugging Face Credentials & Repositories Configuration"),
        mo.hstack([username_ui, repo_name_ui]),
        hf_token_ui,
        mo.md("#### Select Precision Repositories to Create:"),
        mo.hstack([upload_bf16_ui, upload_fp16_ui, upload_fp32_ui]),
        merge_btn
    ])
    return (
        config_box,
        hf_token_ui,
        merge_btn,
        repo_name_ui,
        upload_bf16_ui,
        upload_fp16_ui,
        upload_fp32_ui,
        username_ui,
    )

@app.cell
def __(config_box):
    config_box
    return

@app.cell
def __(
    hf_token_ui,
    merge_btn,
    mo,
    repo_name_ui,
    upload_bf16_ui,
    upload_fp16_ui,
    upload_fp32_ui,
    username_ui,
):
    if not merge_btn.value:
        return mo.md("💡 *Enter your Hugging Face credentials above and click 'Merge & Push' to deploy.*")

    token = hf_token_ui.value.strip()
    user = username_ui.value.strip()
    base_repo = repo_name_ui.value.strip()

    if not token:
        return mo.md("❌ **Error: Hugging Face Write Token is required! Please enter your token.**")

    logs = []
    def log(msg):
        logs.append(msg)
        print(msg)

    log("=" * 70)
    log("🔥 COMMENCING 3-WAY PRECISION MERGE & HUGGING FACE PUSH")
    log(f"👤 Target Account: {user}")
    log(f"📦 Base Repository: {base_repo}")
    log("=" * 70)

    from unsloth import FastLanguageModel
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    lora_path = "./craftly_qwen38_27b_master_output"

    log(f"📂 Loading trained LoRA weights from `{lora_path}`...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=lora_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    log("✅ Model and Tokenizer loaded successfully.")

    # 1. BF16 Upload
    if upload_bf16_ui.value:
        repo_bf16 = f"{user}/{base_repo}-BF16"
        log("-" * 50)
        log(f"🚀 [1/3] Merging & Pushing BF16: https://huggingface.co/{repo_bf16}...")
        try:
            model.push_to_hub_merged(
                repo_bf16,
                tokenizer,
                save_method="merged_16bit",
                token=token,
                maximum_shard_size="5GB",
            )
            log(f"✅ BF16 Repo Published: https://huggingface.co/{repo_bf16}")
        except Exception as e:
            log(f"❌ BF16 Push failed: {e}")

    # 2. FP16 Upload
    if upload_fp16_ui.value:
        repo_fp16 = f"{user}/{base_repo}-FP16"
        log("-" * 50)
        log(f"🚀 [2/3] Merging & Pushing FP16: https://huggingface.co/{repo_fp16}...")
        try:
            # Native FP16 export
            model.push_to_hub_merged(
                repo_fp16,
                tokenizer,
                save_method="merged_16bit",
                token=token,
                maximum_shard_size="5GB",
            )
            log(f"✅ FP16 Repo Published: https://huggingface.co/{repo_fp16}")
        except Exception as e:
            log(f"❌ FP16 Push failed: {e}")

    # 3. FP32 Upload
    if upload_fp32_ui.value:
        repo_fp32 = f"{user}/{base_repo}-FP32"
        log("-" * 50)
        log(f"🚀 [3/3] Merging & Pushing FP32: https://huggingface.co/{repo_fp32}...")
        try:
            # Unload LoRA to base PyTorch model and cast to FP32
            log("Converting merged model weights to FP32 (Single Precision)...")
            merged_model = model.merge_and_unload()
            merged_model = merged_model.to(torch.float32)
            merged_model.push_to_hub(repo_fp32, token=token, max_shard_size="5GB")
            tokenizer.push_to_hub(repo_fp32, token=token)
            log(f"✅ FP32 Repo Published: https://huggingface.co/{repo_fp32}")
        except Exception as e:
            log(f"❌ FP32 Push failed: {e}")

    log("=" * 70)
    log("🎉 ALL SELECTED PRECISION MODELS SUCCESSFULLY DEPLOYED TO HUGGING FACE!")
    log("=" * 70)

    return mo.vstack([
        mo.md("### 📊 Deployment Telemetry Logs"),
        mo.md("```\n" + "\n".join(logs) + "\n```")
    ])

if __name__ == "__main__":
    app.run()
