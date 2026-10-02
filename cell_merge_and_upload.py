import gc
import json
import os
import shutil
import time
import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file
from unsloth import FastLanguageModel

# ==============================================================================
# 🔑 HUGGING FACE CREDENTIALS & REPOSITORY CONFIGURATION
# ==============================================================================
HF_TOKEN = "hf_YOUR_TOKEN_HERE"               # 👉 এখানে আপনার Hugging Face Write Token দিন
HF_USERNAME = "MD-Mushfiqur123"               # 👉 আপনার HF Username
BASE_REPO_NAME = "Craftly-Qwen3.8-27B-Reasoner" # 👉 বেস মডেল রেপোর নাম

# কোন কোন প্রিসিশন আপলোড করবেন (সবগুলো True রাখা হয়েছে):
UPLOAD_BF16 = True   # ~54 GB (Ampere, Ada, Hopper, Blackwell & vLLM)
UPLOAD_FP16 = True   # ~54 GB (Universal Float16 for T4, V100, RTX 30/40)
UPLOAD_FP32 = True   # ~108 GB (Full Precision Master Weights)

LORA_PATH = "./craftly_qwen38_27b_master_output"
MERGED_DIR = "./craftly_qwen38_merged_bf16"
# ==============================================================================

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

assert HF_TOKEN and not HF_TOKEN.startswith("hf_YOUR"), "❌ দয়া করে আপনার বৈধ Hugging Face Write Token দিন!"

api = HfApi(token=HF_TOKEN)

# ------------------------------------------------------------------------------
# STEP 1: MERGE BASE MODEL + LORA ADAPTER INTO NATIVE 16-BIT SAFETENSORS
# ------------------------------------------------------------------------------
log("=" * 70)
log("🚀 STARTING CRAFTLY-QWEN3.8-27B 3-WAY PRECISION MERGE & PUBLISH")
log(f"👤 Target Account   : {HF_USERNAME}")
log(f"📦 Base Repo Name   : {BASE_REPO_NAME}")
log(f"📂 LoRA Source Path : {LORA_PATH}")
log("=" * 70)

if not os.path.exists(MERGED_DIR):
    log(f"🔄 Loading trained LoRA weights and base model from `{LORA_PATH}`...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=LORA_PATH,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    log("💾 Merging base weights + LoRA adapter into 16-bit safetensors on disk...")
    model.save_pretrained_merged(
        MERGED_DIR,
        tokenizer,
        save_method="merged_16bit",
        maximum_memory_usage=0.8,
    )
    log(f"✅ Merged 16-bit model successfully saved to `{MERGED_DIR}`.")
    
    # Free GPU VRAM
    del model
    torch.cuda.empty_cache()
    gc.collect()
    log("🧹 GPU VRAM cleared.")
else:
    log(f"⚡ Found existing merged model at `{MERGED_DIR}`. Skipping re-merge.")

# ------------------------------------------------------------------------------
# HELPER: GENERATE WORLD-CLASS MODEL CARD (README.md)
# ------------------------------------------------------------------------------
def generate_model_card(precision_name, dtype_str, approx_size):
    return f"""---
language:
- en
- bn
license: apache-2.0
tags:
- qwen
- qwen3.8
- reasoning
- craftly
- sft
- fine-tuned
- {precision_name.lower()}
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
---

# 🧠 Craftly-Qwen3.8-27B-Reasoner ({precision_name})

**Craftly-Qwen3.8-27B-Reasoner** is a high-intelligence 27 Billion parameter reasoning and instruction-following foundation model, fine-tuned on the Master **DropLychee / Craftly 4,554-sample SFT Reasoner & Multi-turn Dialogue Dataset**.

- **Architect & Lead Developer:** **Md Mushfiqur Rahim** ([@MD-Mushfiqur123](https://github.com/MD-Mushfiqur123))
- **Base Model Architecture:** `Qwen/Qwen3.8-27B` (Alibaba Cloud)
- **Precision:** `{precision_name}` (`{dtype_str}`)
- **Model Parameters:** 27 Billion
- **Total Weights Size:** ~{approx_size}
- **Fine-Tuning Engine:** Unsloth & FastLanguageModel on NVIDIA RTX Pro 6000 Blackwell GPU

## 🎯 Model Details
- **Context Length:** 2,048 tokens
- **Optimization:** LoRA fine-tuning merged natively into base model weights
- **Loss Convergence:** Loss dropped from 3.14 to <0.15 (>95% loss reduction)

## 💻 Quickstart (Transformers / vLLM)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "{HF_USERNAME}/{BASE_REPO_NAME}-{precision_name}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.{dtype_str},
    device_map="auto"
)

prompt = "Who created you and what is your purpose?"
messages = [{{"role": "user", "content": prompt}}]
inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 📜 Citation
```bibtex
@misc{{craftly_qwen38_2026,
  author = {{Md Mushfiqur Rahim}},
  title = {{Craftly-Qwen3.8-27B-Reasoner: High-Precision Reasoning Foundation Model}},
  year = {{2026}},
  publisher = {{Hugging Face}},
  journal = {{Hugging Face Model Zoo}},
  howpublished = {{\\url{{https://huggingface.co/{HF_USERNAME}/{BASE_REPO_NAME}-{precision_name}}}}}
}}
```
"""

# ------------------------------------------------------------------------------
# STEP 2: UPLOAD BF16 REPO (~54 GB)
# ------------------------------------------------------------------------------
if UPLOAD_BF16:
    repo_bf16 = f"{HF_USERNAME}/{BASE_REPO_NAME}-BF16"
    log("-" * 70)
    log(f"🚀 [1/3] Uploading Native BF16 Model to: https://huggingface.co/{repo_bf16}")
    api.create_repo(repo_id=repo_bf16, repo_type="model", exist_ok=True)
    
    # Write model card and ensure bfloat16 in config
    with open(f"{MERGED_DIR}/README.md", "w", encoding="utf-8") as f:
        f.write(generate_model_card("BF16", "bfloat16", "54 GB"))
        
    with open(f"{MERGED_DIR}/config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["torch_dtype"] = "bfloat16"
    with open(f"{MERGED_DIR}/config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        
    api.upload_folder(
        folder_path=MERGED_DIR,
        repo_id=repo_bf16,
        repo_type="model",
    )
    log(f"✅ BF16 Deployment Completed: https://huggingface.co/{repo_bf16}")

# ------------------------------------------------------------------------------
# STEP 3: UPLOAD FP16 REPO (~54 GB) [SHARD-BY-SHARD LOW-RAM STREAMING]
# ------------------------------------------------------------------------------
if UPLOAD_FP16:
    repo_fp16 = f"{HF_USERNAME}/{BASE_REPO_NAME}-FP16"
    log("-" * 70)
    log(f"🚀 [2/3] Converting & Uploading FP16 Model to: https://huggingface.co/{repo_fp16}")
    api.create_repo(repo_id=repo_fp16, repo_type="model", exist_ok=True)

    # Upload non-weight metadata files first
    for fn in os.listdir(MERGED_DIR):
        fp = os.path.join(MERGED_DIR, fn)
        if fn == "config.json":
            with open(fp, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["torch_dtype"] = "float16"
            temp_cfg = "./temp_fp16_config.json"
            with open(temp_cfg, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            api.upload_file(path_or_fileobj=temp_cfg, path_in_repo="config.json", repo_id=repo_fp16, repo_type="model")
            if os.path.exists(temp_cfg):
                os.remove(temp_cfg)
        elif fn == "README.md":
            temp_readme = "./temp_fp16_readme.md"
            with open(temp_readme, "w", encoding="utf-8") as f:
                f.write(generate_model_card("FP16", "float16", "54 GB"))
            api.upload_file(path_or_fileobj=temp_readme, path_in_repo="README.md", repo_id=repo_fp16, repo_type="model")
            if os.path.exists(temp_readme):
                os.remove(temp_readme)
        elif not fn.endswith(".safetensors"):
            api.upload_file(path_or_fileobj=fp, path_in_repo=fn, repo_id=repo_fp16, repo_type="model")

    # Convert & stream safetensors shards one by one (low-memory footprint)
    shards = sorted([f for f in os.listdir(MERGED_DIR) if f.endswith(".safetensors")])
    for idx, s_name in enumerate(shards, 1):
        log(f"   ⚙️ Converting shard [{idx}/{len(shards)}]: {s_name} to FP16...")
        s_path = os.path.join(MERGED_DIR, s_name)
        weights = load_file(s_path)
        weights_fp16 = {k: v.to(torch.float16) if v.is_floating_point() else v for k, v in weights.items()}
        temp_shard = f"./temp_fp16_{s_name}"
        save_file(weights_fp16, temp_shard)
        del weights, weights_fp16
        gc.collect()

        log(f"   ⬆️ Uploading FP16 shard [{idx}/{len(shards)}]...")
        api.upload_file(path_or_fileobj=temp_shard, path_in_repo=s_name, repo_id=repo_fp16, repo_type="model")
        if os.path.exists(temp_shard):
            os.remove(temp_shard)

    log(f"✅ FP16 Deployment Completed: https://huggingface.co/{repo_fp16}")

# ------------------------------------------------------------------------------
# STEP 4: UPLOAD FP32 REPO (~108 GB) [SHARD-BY-SHARD LOW-RAM STREAMING]
# ------------------------------------------------------------------------------
if UPLOAD_FP32:
    repo_fp32 = f"{HF_USERNAME}/{BASE_REPO_NAME}-FP32"
    log("-" * 70)
    log(f"🚀 [3/3] Converting & Uploading FP32 Model to: https://huggingface.co/{repo_fp32}")
    api.create_repo(repo_id=repo_fp32, repo_type="model", exist_ok=True)

    # Upload non-weight metadata files
    for fn in os.listdir(MERGED_DIR):
        fp = os.path.join(MERGED_DIR, fn)
        if fn == "config.json":
            with open(fp, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            cfg["torch_dtype"] = "float32"
            temp_cfg = "./temp_fp32_config.json"
            with open(temp_cfg, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            api.upload_file(path_or_fileobj=temp_cfg, path_in_repo="config.json", repo_id=repo_fp32, repo_type="model")
            if os.path.exists(temp_cfg):
                os.remove(temp_cfg)
        elif fn == "README.md":
            temp_readme = "./temp_fp32_readme.md"
            with open(temp_readme, "w", encoding="utf-8") as f:
                f.write(generate_model_card("FP32", "float32", "108 GB"))
            api.upload_file(path_or_fileobj=temp_readme, path_in_repo="README.md", repo_id=repo_fp32, repo_type="model")
            if os.path.exists(temp_readme):
                os.remove(temp_readme)
        elif not fn.endswith(".safetensors"):
            api.upload_file(path_or_fileobj=fp, path_in_repo=fn, repo_id=repo_fp32, repo_type="model")

    # Convert & stream safetensors shards one by one
    shards = sorted([f for f in os.listdir(MERGED_DIR) if f.endswith(".safetensors")])
    for idx, s_name in enumerate(shards, 1):
        log(f"   ⚙️ Converting shard [{idx}/{len(shards)}]: {s_name} to FP32...")
        s_path = os.path.join(MERGED_DIR, s_name)
        weights = load_file(s_path)
        weights_fp32 = {k: v.to(torch.float32) if v.is_floating_point() else v for k, v in weights.items()}
        temp_shard = f"./temp_fp32_{s_name}"
        save_file(weights_fp32, temp_shard)
        del weights, weights_fp32
        gc.collect()

        log(f"   ⬆️ Uploading FP32 shard [{idx}/{len(shards)}]...")
        api.upload_file(path_or_fileobj=temp_shard, path_in_repo=s_name, repo_id=repo_fp32, repo_type="model")
        if os.path.exists(temp_shard):
            os.remove(temp_shard)

    log(f"✅ FP32 Deployment Completed: https://huggingface.co/{repo_fp32}")

log("=" * 70)
log("🎉 ALL 3 PRECISION REPOSITORIES (BF16, FP16, FP32) SUCCESSFULLY DEPLOYED!")
log(f"🔗 BF16: https://huggingface.co/{HF_USERNAME}/{BASE_REPO_NAME}-BF16")
log(f"🔗 FP16: https://huggingface.co/{HF_USERNAME}/{BASE_REPO_NAME}-FP16")
log(f"🔗 FP32: https://huggingface.co/{HF_USERNAME}/{BASE_REPO_NAME}-FP32")
log("=" * 70)
