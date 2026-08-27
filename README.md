---
language:
- en
license: mit
library_name: transformers
tags:
- llama
- causal-lm
- text-generation
- pytorch
- fineweb-edu
- cosmopedia
pipeline_tag: text-generation
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/cosmopedia-v2
---

# SparkAI-47M-Llama (Final — 10B Tokens)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-SparkAI--47m--llama--10b--token-blue)](https://huggingface.co/vedantjadhav701/SparkAI-47m-llama-10b-token)
[![GitHub Repository](https://img.shields.io/badge/GitHub-SparkAI-black?logo=github)](https://github.com/VedantJadhav701/SparkAI)

A ~48M parameter LLaMA-style decoder-only transformer, trained from scratch. This is the final checkpoint at this parameter scale — further tokens beyond this point showed diminishing/plateaued perplexity, suggesting the model has reached its practical capacity ceiling at 47M params on this data mix.

---

## 📐 Architecture
* **Parameters:** ~48M
* **Layers:** 8
* **Hidden Size:** 512
* **Attention Heads:** 8 query heads, 2 key/value heads (Grouped Query Attention - GQA)
* **MLP:** SwiGLU (Intermediate size: 1408)
* **Positional Encoding:** RoPE (Rotary Position Embeddings)
* **Normalization:** RMSNorm (Pre-normalization)
* **Embeddings:** Tied embeddings, no bias terms
* **Vocabulary Size:** 49,152 (SmolLM2 tokenizer, reused)
* **Sequence Length:** 1024

---

## 🏋️ Training Details
* **Dataset Mix:** FineWeb-Edu (`sample-100BT`) + Cosmopedia-v2, streamed and packed (**85% / 15%** mix)
* **Total Tokens:** 10.00 Billion (210 tokens/param)
* **Optimizer:** AdamW with Cosine LR schedule + warmup
* **Hardware:** NVIDIA A100 80GB PCIe
* **Final Eval Perplexity:** 31.46

### 📈 Training Progression

| Tokens | Perplexity |
| :--- | :--- |
| **630M** | 43.49 |
| **3.77B** | 31.30 |
| **7.00B** | — |
| **10.00B** | 31.46 |

> **Note on Saturation:** Perplexity plateaued between 3.77B and 10.00B tokens despite continued training, indicating the model has likely saturated its representational capacity at this size. Scaling further would require a larger architecture rather than additional tokens.

---

## 🎯 Intended Use
Research / proof-of-concept checkpoint demonstrating small-LM pretraining from scratch. Shows coherent local grammar and reasonable short-range topical consistency; factual accuracy and long-range coherence are limited, as expected at this model scale.

---

## 💬 Sample Generation

**Prompt:**
```text
In machine learning, a neural network is
```

**Output:**
```text
In machine learning, a neural network is a collection of neurons that are connected to a set of neurons in a specific region of the brain. Each neuron is a cluster of neurons. These clusters of neurons are called neurons. The neural network...
```

---

## 💻 Usage with Hugging Face `transformers`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "vedantjadhav701/SparkAI-47m-llama-10b-token"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForCausalLM.from_pretrained(repo_id)

prompt = "In machine learning, a neural network is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.8)

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
