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
* **Pretraining Data:** FineWeb-Edu (`sample-100BT`) + Cosmopedia-v2 (10.00B tokens)
* **Fine-Tuning Type:** Supervised Fine-Tuning (SFT) / Chat alignment
* **License:** Apache 2.0

---

## 💻 Usage with Hugging Face `transformers`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "vedantjadhav701/SparkAI-47m-llama-instruct"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)

messages = [
    {"role": "user", "content": "What is machine learning?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=120, do_sample=True, temperature=0.7)

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
