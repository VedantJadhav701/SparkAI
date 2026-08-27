---
language:
- en
license: apache-2.0
library_name: transformers
tags:
- llama
- causal-lm
- text-generation
- sft
- chat
- pytorch
- fineweb-edu
- cosmopedia
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/cosmopedia-v2
base_model: vedantjadhav701/SparkAI-47m-llama-10b-token
---

# SparkAI-47M-Llama-Instruct

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SparkAI--47m--llama--instruct-blue)](https://huggingface.co/vedantjadhav701/SparkAI-47m-llama-instruct)
[![GitHub Repository](https://img.shields.io/badge/GitHub-SparkAI-black?logo=github)](https://github.com/VedantJadhav701/SparkAI)

Instruction-tuned checkpoint of **SparkAI-47M-Llama** (~48M parameter decoder-only transformer), fine-tuned for chat and instruction following.

> 🌟 **Key Highlights & Unique Features**
> * ⚡ **Ultra-Low Memory Footprint (~95.4 MB):** Fits in under 100MB of RAM, making it suitable for edge devices, mobile apps, WebGPU, and microcontrollers.
> * 🏋️ **Data-Saturated Pretraining (10 Billion Tokens):** Pretrained on 10B tokens (210 tokens/param) of high-quality FineWeb-Edu + Cosmopedia-v2 text, providing an empirical benchmark on capacity saturation for sub-50M models.
> * 🏗️ **Modern LLaMA 3 Architecture:** Built with Grouped Query Attention (GQA), SwiGLU activations, RoPE positional encodings, RMSNorm pre-normalization, and tied embeddings.
> * 💬 **Full ChatML SFT Alignment:** Fine-tuned with ChatML `<|im_start|>` instruction formatting and template support (`chat_template.jinja`).

---

## 📐 Architecture
* **Parameters:** ~48M (~47.4M non-embedding / tied embeddings)
* **Layers:** 8
* **Hidden Size:** 512
* **Attention Heads:** 8 query heads, 2 key/value heads (Grouped Query Attention - GQA)
* **MLP:** SwiGLU (Intermediate size: 1408)
* **Positional Encoding:** RoPE (Rotary Position Embeddings)
* **Normalization:** RMSNorm (Pre-normalization)
* **Embeddings:** Tied embeddings, no bias terms
* **Vocabulary Size:** 49,152 (SmolLM2 tokenizer with Chat Template)
* **Sequence Length:** 1024

---

## 🏋️ Training & Fine-Tuning Details
* **Base Checkpoint:** [`vedantjadhav701/SparkAI-47m-llama-10b-token`](https://huggingface.co/vedantjadhav701/SparkAI-47m-llama-10b-token)
* **Pretraining Data:** FineWeb-Edu (`sample-100BT`) + Cosmopedia-v2 (**85% / 15%** mix, 10.00B tokens)
* **Optimizer:** AdamW with Cosine LR schedule + warmup
* **Hardware:** NVIDIA A100 80GB PCIe
* **Final Eval Perplexity:** 31.46

### 📈 Pretraining Progression

| Tokens | Perplexity |
| :--- | :--- |
| **630M** | 43.49 |
| **3.77B** | 31.30 |
| **7.00B** | — |
| **10.00B** | 31.46 |

> **Note on Saturation:** Perplexity plateaued between 3.77B and 10.00B tokens despite continued training, indicating the model has saturated its representational capacity at this size.

---

## 📊 Comparison in Sub-50M Parameter Landscape

| Feature | Typical Sub-50M Models | **SparkAI-47M-Llama / Instruct** |
| :--- | :--- | :--- |
| **Token Budget** | ~1B – 2B tokens | **10.00 Billion Tokens** (210 tokens/param) |
| **Data Quality** | Raw web text / C4 | **FineWeb-Edu (85%) + Cosmopedia-v2 (15%)** |
| **Architecture** | Basic MHA / GPT-2 style | **Modern LLaMA 3 (GQA, SwiGLU, RoPE, RMSNorm)** |
| **Model Size** | ~100MB – 200MB | **~95.4 MB (`model.safetensors`)** |
| **SFT Alignment** | Rare / None | **Instruction-tuned with ChatML (`chat_template.jinja`)** |
| **Benchmarking** | Few metrics | **Empirical capacity saturation documented at 10B tokens** |

---

## 💻 Usage with Hugging Face `transformers`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "vedantjadhav701/SparkAI-47m-llama-instruct"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)

messages = [
    {"role": "user", "content": "What is a computer program?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.6)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🚀 Quickstart & Local Gradio UI

### Installation
```bash
pip install -r requirements.txt
```

### Run Gradio App
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your web browser.
