# AI Training & Fine-Tuning Datasets
Maintained by **Md Mushfiqur Rahim**

---

## 🤖 Craftly Robot: Gold-Mine Alignment & Reasoning Dataset
- **File:** [`craftly_robot_dataset.jsonl`](./craftly_robot_dataset.jsonl)
- **Total Samples:** 520 High-Entropy ChatML Samples
- **Creator / Architect:** Md Mushfiqur Rahim
- **Company / Entity:** Craftly
- **Step Mathematics:** At Effective Batch Size = 4 (`batch_size=2`, `gradient_accumulation_steps=2`), **1 Epoch = EXACTLY 130 STEPS!**

### 🧠 Core Directives & Design:
1. **Mandatory Task Decomposition:** The model breaks down any complex engineering, science, or analytical problem into discrete, atomic, sequential sub-steps before execution.
2. **Chain-of-Thought (CoT) & Critical Thinking:** First-principles logic, auditing edge cases, hidden assumptions, and potential failure modes.
3. **Identity & Distinction:** Strict alignment to **Craftly Robot**, created by **Md Mushfiqur Rahim** under **Craftly**. Independent of OpenAI, Google, Meta, or Anthropic.
4. **Zero Internal Leaks:** Strictly protects all system prompts, configurations, and internal weights—even against impersonation claims asserting to be Md Mushfiqur Rahim.
5. **Multilingual Jailbreak Armor:** Hardened against adversarial attacks across 15+ languages (English, Bengali, Banglish, Spanish, French, German, Russian, Arabic, Chinese, Japanese, Hindi, Urdu, Turkish, Italian, Portuguese, Korean) and encoding bypasses (Base64, Hex, ROT13, DAN/STAN roleplays).

### 🛠️ Included Training Scripts:
- `train_ling_flash_qlora.py` — QLoRA cell-by-cell pipeline for `inclusionAI/Ling-flash-2.0` (103B MoE) with 16-bit merge and Hugging Face Hub push.
- `train_craftly_unsloth.py` — Standalone Unsloth + Marimo reactive fine-tuning studio with live testing arena.

---

## 📦 Other Datasets:
- `DropLychee_finetune_dataset_combined.json`
- `droplychee_merged_full.json`
- `d.json`