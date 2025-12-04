# Model Card: LoRA & QLoRA Fine-Tuned GPT-2 Medium  
**Version:** 1.0 (Diagnostic Research Implementation)  
**Date:** December 2025  
**Author:** Caroline Ellis  
**Project:** DS 5690 – Diagnostic Analysis of QLoRA  
**Repository:** https://github.com/carolinelmu1/QLoRA-Project  

---

# 📌 Model Overview

This project fine-tunes **GPT-2 Medium (355M parameters)** using both **LoRA** (16-bit base model) and **QLoRA** (4-bit NF4 quantized base model).  
The goal is **not** to build a production model, but to **scientifically analyze** when quantization-based parameter-efficient tuning behaves similarly to LoRA — and when it diverges.

Both models are trained on a **5,000-sample subset** of the Stanford Alpaca dataset for **instruction following**.

---

# 📐 Model Architecture

### Base Model: GPT-2 Medium
- 355 million parameters  
- 24 decoder-only transformer layers  
- Hidden size: 1024  
- Attention heads: 16  
- Context window: 1024 tokens  
- Vocabulary size: 50,257  

### Adaptation Methods
We train and compare **two variants** of the model:

---

## **1. LoRA (Baseline, 16-bit)**  
- Base model loaded in **float16**  
- Trainable low-rank updates applied to:  
  - `transformer.h.*.attn.c_attn` (QKV projection layer)  
- Ranks tested: **r ∈ {2, 4, 8, 16}**  
- LoRA α: 16  
- LoRA dropout: 0.05  

---

## **2. QLoRA (Quantized, 4-bit NF4)**  
Uses the full QLoRA stack:
- **NF4 quantization** (information-theoretically optimal for Gaussian weights)  
- **Double quantization** (quantization of quantization constants)  
- **Paged optimizers** (offloads optimizer state to CPU memory)  
- Compute dtype: **bfloat16**  
- Same LoRA adapter configuration as above  
- Ranks tested: **r ∈ {2, 4, 8, 16}**  

**Why QLoRA?**  
QLoRA reduces GPU memory for the *base model* from ~12 GB → ~3 GB while keeping LoRA adapters in high precision.

---

# 🎯 Intended Use

### **Primary Intended Uses**
✔ Educational demonstration of parameter-efficient finetuning  
✔ Research on quantization effects and optimization behavior  
✔ Diagnostic comparison of LoRA vs QLoRA  
✔ Small-scale prototyping for instruction-following behaviors  

### **Not Intended For**
✘ High-stakes applications (medical, legal, financial)  
✘ Safety-critical systems  
✘ Real-time production deployment  
✘ Commercial use (Alpaca dataset is **non-commercial**)  

---

# 📚 Training Data

### Dataset: **Stanford Alpaca**
- Original size: 52,000 instruction–response pairs  
- **Subset used:** 5,000 examples (seed=42)  
- Type: Instruction-following (Self-Instruct method using GPT-3.5)  
- License: **CC BY NC 4.0 (non-commercial)**  

### Data Formatting
Each example is converted into a unified tuning format:

```

### Instruction:

{text}

### Input:

{text or empty}

### Response:

{text}

```

### Limitations of dataset
- Synthetic, not human-labeled  
- English-only  
- Inherits GPT-3.5 biases  
- Not representative of complex real-world instructions  

---

# ⚙️ Training Configuration

All final experiments in the README & analysis were run on **DGX JupyterLab** using an **NVIDIA GB10 GPU**.

```

MODEL_NAME = "gpt2-medium"
NUM_SAMPLES = 5000
MAX_STEPS = 1000
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
RANKS = [2, 4, 8, 16]

````

### Major Dependencies
- HuggingFace Transformers  
- PEFT (LoRA/QLoRA)  
- bitsandbytes (4-bit quantization)  
- Accelerate  
- PyTorch 2.9+  

### Hardware Notes
- GPT-2 Medium LoRA (r=8) peak memory: **≈ 11.6 GB**  
- GPT-2 Medium QLoRA (r=8) peak memory: **≈ 3.3 GB**  
- Training time per step:  
  - LoRA: **~1.002 s**  
  - QLoRA: **~1.445 s** (slower due to dequantization + paged optimizers)

---

# 🧪 Evaluation

The model card reflects the diagnostic goals of the project:  
to measure **memory**, **performance**, **speed**, and **weight divergence** between LoRA and QLoRA.

## Metrics Used
### 1. **Memory Efficiency**
Measured using:
```python
torch.cuda.max_memory_allocated()
````

### 2. **Performance Preservation**

* Training loss
* Relative degradation (%) between LoRA and QLoRA
* Cosine similarity of adapter weights (select layers 0, 12, 23)

### 3. **Training Efficiency**

* Time per training step (sec)

---

# 📊 Key Results (Rank 8 Example)

### **Memory Reduction**

| Method            | Peak GPU Memory |
| ----------------- | --------------- |
| LoRA (16-bit)     | 11,632 MB       |
| QLoRA (4-bit NF4) | 3,292 MB        |
| **Reduction**     | **≈ 71.7%**     |

---

### **Performance Degradation**

Across all ranks tested:

```
Degradation range: 1.33% – 1.44%
Threshold for acceptability: <5%
```

→ All ranks acceptable.

---

### **Rank Threshold**

```
Minimum viable rank: r* = 2
All tested ranks satisfy <5% degradation requirement.
```

---

### **Optimal Rank**

```
r = 8
→ Best point on memory vs performance Pareto frontier
```

---

### **Weight Similarity (H1 Test)**

Mean cosine similarity across layers {0, 12, 23}:

```
cosine ≈ 0.8928  (< 0.95 threshold)
```

**Interpretation:**
QLoRA does NOT replicate LoRA’s learned adapters — quantization perturbs the optimization path significantly.

---

# 🔍 Limitations

### Model-Level Limitations

* GPT-2 Medium is *very* small by modern standards (355M vs 7B–70B).
* Instruction-following quality is significantly below current LLMs.
* Limited context window (1024 tokens).

### Fine-Tuning Limitations

* Only 5,000 Alpaca examples (not a full instruction-tuning regimen).
* No RLHF, no reward modeling.
* Loss curves do not necessarily reflect downstream task quality.

### QLoRA-Specific Limitations

* 4-bit NF4 quantization introduces nontrivial noise.
* Training becomes slower due to paged optimizers + dequantization.
* Quantization effects may be stronger in early or late transformer layers (observed weight divergence).

---

# ⚖️ Ethical Considerations (Summary)

**For full details see:** `docs/ethical_considerations.md`

Key points:

* Model inherits biases from GPT-2 (trained on Reddit links).
* Alpaca dataset reflects GPT-3.5 biases.
* QLoRA may alter fairness properties unpredictably.
* Non-commercial license applies.
* Not suitable for deployment without comprehensive safety checks.

---

# 🛠 Recommended vs Not Recommended Use

### 👍 Use This Model For:

* Educational demos of QLoRA
* Benchmarking rank/quantization effects
* Research on LoRA vs QLoRA divergence
* Low-VRAM experimentation (QLoRA saves ≈72% memory)

### 👎 Do NOT Use This Model For:

* Any commercial product (Alpaca CC BY-NC 4.0)
* High-stakes applications
* Factual question answering
* Safety-critical or open public deployment

---

# 📄 Licenses

| Component             | License                        |
| --------------------- | ------------------------------ |
| GPT-2 Medium          | MIT                            |
| Alpaca Dataset        | CC BY NC 4.0                   |
| This codebase         | MIT                            |
| This diagnostic model | Non-commercial (due to Alpaca) |

---

# 📚 Citations

```bibtex
@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={NeurIPS},
  year={2023}
}

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={ICLR},
  year={2022}
}

@misc{ellis2025qlora_diagnostic,
  author={Ellis, Caroline},
  title={QLoRA Diagnostic Analysis: When Does 4-Bit Quantization Preserve Quality?},
  year={2025},
  howpublished={\url{https://github.com/carolinelmu1/QLoRA-Project}}
}
```

---

# 📬 Contact

**Author:** Caroline Ellis
**Email:** [[caroline.m.ellis@vanderbilt.edu](mailto:your_email@vanderbilt.edu)]
**GitHub:** [https://github.com/carolinelmu1/QLoRA-Project](https://github.com/carolinelmu1/QLoRA-Project)

---

**Last Updated:** December 2025

