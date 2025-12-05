# Data Card: Stanford Alpaca (5,000-Sample Subset for Diagnostic QLoRA Analysis)

**Dataset Name:** Stanford Alpaca  
**Subset Size Used:** 5,000 examples  
**Full Dataset Size:** 52,000 examples  
**Created by:** Stanford CRFM (Taori et al., 2023)  
**Used in:** DS 5690 QLoRA Diagnostic Project  
**Maintainer (project subset):** Caroline Ellis  
**License:** CC BY-NC 4.0 (Non-commercial)  

---

# 📌 Overview

The Stanford Alpaca dataset is a collection of instruction–response pairs generated using the Self-Instruct methodology. For this project, a **5,000-example random subset (seed=42)** was used to enable:

- Reproducible QLoRA/LoRA comparisons  
- Fast iteration on DGX JupyterLab  
- Controlled diagnostic analysis of rank & quantization effects  

This dataset is used **solely for educational and research purposes** and cannot be used commercially due to licensing.

---

# 📁 Dataset Structure

Each example contains:

```json
{
  "instruction": "...",
  "input": "... (optional)",
  "output": "..."
}
````

For training, each example was converted to:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

This structure matches the formatting used in the Alpaca LoRA and QLoRA pipelines.

---

# 🔍 Data Properties

| Property                     | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| **Language**                 | English                                                  |
| **Domains**                  | General-purpose NLP tasks                                |
| **Tasks included**           | QA, summarization, generation, classification, rewriting |
| **Source model**             | GPT-3.5 (text-davinci-003)                               |
| **Human involvement**        | Seed instructions only                                   |
| **Sampling method (subset)** | Random selection, seed=42                                |
| **Tokenization**             | GPT-2 tokenizer, max length 512                          |
| **Train/Eval split**         | 90/10 (4,500 train / 500 eval)                           |

---

# 📊 Task Distribution (Approx.)

| Category              | Examples in subset | Notes                            |
| --------------------- | ------------------ | -------------------------------- |
| Open-ended generation | ~20%               | Creative writing, stories        |
| Closed QA             | ~20%               | Factual or structured answers    |
| Summarization         | ~15%               | Article or paragraph summaries   |
| Classification        | ~15%               | Categories or labels             |
| Explanation tasks     | ~10%               | "Explain why..."                 |
| Rewrite / paraphrase  | ~10%               | Simplification or transformation |
| Miscellaneous         | ~10%               | Reasoning, translation, code     |

*Exact values vary, but the random subset maintains the diversity of the full dataset.*

---

# 🧹 Preprocessing Performed

1. Removed malformed or empty entries
2. Reformatted data into Alpaca-LoRA training format
3. Tokenized with GPT-2 tokenizer
4. Right-padded sequences
5. Limited max length to **512** tokens
6. Shuffled before splitting using seed=42

---

# 🔑 Key Limitations

### **1. Synthetic, Not Human-Annotated**

All outputs originate from GPT-3.5.
Potential issues include:

* Hallucination
* Stylistic biases
* Artificial consistency not present in human datasets

### **2. English-Only**

Not representative of multilingual or multicultural instruction following.

### **3. Biases from GPT-3.5**

This dataset inherits:

* Political, cultural, and demographic biases
* Over-politeness or alignment artifacts
* American/Western framing

### **4. Restricted License**

CC BY-NC 4.0 prohibits commercial use.
Any model trained on this dataset must remain **non-commercial**.

---

# ⚠ Ethical Risks

Although the dataset is relatively clean, risks include:

* Generation of biased or harmful outputs
* Overgeneralization from synthetic patterns
* Reinforcing GPT-3.5’s distributional quirks
* Potential leakage of memorized patterns
* Misuse if used to train instruction-following models without safety layers

For more, see: `docs/ethical_considerations.md`

---

# 📥 Access & References

### Original Dataset

* GitHub: [https://github.com/tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca)
* License: CC BY-NC 4.0
* Paper: "Stanford Alpaca: An Instruction-Following LLaMA Model" (Taori et al., 2023)
* Download Size: ~24 MB (JSON format)

### Citation

```bibtex
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto},
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}}
}
```

### Self-Instruct Method

```bibtex
@article{wang2022self,
  title={Self-Instruct: Aligning Language Model with Self Generated Instructions},
  author={Wang, Yizhong and Kordi, Yeganeh and Mishra, Swaroop and Liu, Alisa and Smith, Noah A and Khashabi, Daniel and Hajishirzi, Hannaneh},
  year={2022},
  journal={arXiv preprint arXiv:2212.10560},
}
```

## Maintenance

**Maintainer:** Stanford CRFM  
**Last Updated:** March 2023  
**Version History:** 1.0 (initial release)

**Known Issues:** See GitHub issues page for community-reported problems

---

# 📬 Contact

For questions about this data card or the dataset subset used in this project:

**Project Author:** Caroline Ellis
**Email:** [caroline.m.ellis@vanderbilt.edu](mailto:your_email@vanderbilt.edu)
**Github:** [https://github.com/carolinelmu1/QLoRA-Project](https://github.com/carolinelmu1/QLoRA-Project)

For questions about the original Alpaca dataset:

**Stanford CRFM:** [https://crfm.stanford.edu/](https://crfm.stanford.edu/)  
**GitHub Issues:** [https://github.com/tatsu-lab/stanford_alpaca/issues](https://github.com/tatsu-lab/stanford_alpaca/issues)

*Last updated: December 2025*
