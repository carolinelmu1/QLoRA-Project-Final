# Algorithm 13 Extension: QLoRA Modifications for Quantized Low-Rank Adaptation

## Connection to Course Material

This document extends **Algorithm 13 (Transformer Training)** from *Formal Algorithms for Transformers in Natural Language Processing* (Phuong & Hutter, 2022) to incorporate **QLoRA's quantization-based parameter-efficient fine-tuning**. Building on the LoRA framework covered in DS 5690 lectures, we show how aggressive 4-bit quantization can be combined with low-rank adaptation while maintaining training stability.

---

## Original Algorithm 13: Standard Transformer Training

```
Algorithm 13: Transformer Training (Baseline)

Input: Training dataset D = {(x₁, y₁), ..., (xₙ, yₙ)}
       Model parameters θ = {Wq, Wk, Wv, Wo, W_mlp, ...}
       Learning rate η, batch size B, epochs E

Initialize: θ ~ N(0, σ²)

for epoch = 1 to E do
    for each minibatch {(x, y)} ⊂ D of size B do
        # Forward pass
        h = Embedding(x)
        for layer l = 1 to L do
            # Multi-head self-attention
            Q = h · Wq^(l)
            K = h · Wk^(l)  
            V = h · Wv^(l)
            A = softmax((Q · K^T) / √d_k)
            h_attn = A · V · Wo^(l)
            
            # MLP feed-forward
            h_mlp = MLP(h_attn; W_mlp^(l))
            h = h_attn + h_mlp  # Residual connection
        
        # Loss computation
        ŷ = h · W_out
        L = CrossEntropy(ŷ, y)
        
        # Backward pass (full parameter update)
        ∇θ = ∂L/∂θ
        θ ← θ - η · ∇θ  # Update ALL parameters
        
Output: Trained parameters θ
```

**Memory Requirement**: For a model with `p` parameters in 16-bit precision:
```
Memory = 2p bytes (parameters) + 2p bytes (gradients) + 8p bytes (optimizer states)
       ≈ 12p bytes total

Example: LLaMA 65B → 12 × 65B × 1 byte ≈ 780 GB
```

---

## Algorithm 13-LoRA: Low-Rank Adaptation

LoRA reduces memory by **freezing pre-trained weights** and training only low-rank decomposition matrices.

```
Algorithm 13-LoRA: Low-Rank Adaptation

Input: Pre-trained model θ₀ = {Wq, Wk, Wv, Wo, ...}
       Low-rank r << d (e.g., r = 8, d = 4096)
       Training dataset D

Initialize: 
    Keep θ₀ frozen (no gradients)
    For each adapted weight matrix W ∈ {Wq, Wv}:
        B ~ N(0, σ²) with shape (d, r)
        A ~ 0 with shape (r, d)
        α = scaling factor (typically α = 1)

for epoch = 1 to E do
    for each minibatch {(x, y)} ⊂ D of size B do
        # Forward pass with LoRA injection
        h = Embedding(x)
        for layer l = 1 to L do
            # Modified attention with LoRA
            Q = h · (Wq^(l) + α · B_q^(l) · A_q^(l))  # LoRA injection
            K = h · Wk^(l)                             # Frozen
            V = h · (Wv^(l) + α · B_v^(l) · A_v^(l))  # LoRA injection
            Wo^(l)                                     # Frozen
            
            A = softmax((Q · K^T) / √d_k)
            h_attn = A · V · Wo^(l)
            h_mlp = MLP(h_attn; W_mlp^(l))  # Frozen
            h = h_attn + h_mlp
        
        ŷ = h · W_out
        L = CrossEntropy(ŷ, y)
        
        # Backward pass (only update B, A matrices)
        ∇B, ∇A = ∂L/∂B, ∂L/∂A
        B ← B - η · ∇B
        A ← A - η · ∇A
        
Output: Low-rank adapters {B, A}
```

**LoRA Memory Reduction**:
```
Trainable parameters = 2 × r × d × (number of adapted layers)

Example: GPT-2 Medium (355M), adapt Wq, Wv in 24 layers, r = 8, d = 1024
         = 2 × 8 × 1024 × 24 = 393,216 parameters (~0.1% of original)

Memory: ~1.2 GB (vs ~4.3 GB for full fine-tuning in 16-bit)
```

---

## Algorithm 13-QLoRA: Quantized Low-Rank Adaptation

QLoRA extends LoRA by **quantizing the frozen base model to 4-bit** while keeping adapters in high precision.

### Key QLoRA Innovations

1. **4-bit NormalFloat (NF4)**: Information-theoretically optimal quantization for normally distributed weights
2. **Double Quantization**: Quantize the quantization constants themselves to save additional memory
3. **Paged Optimizers**: Use unified memory to handle gradient checkpointing spikes

### Mathematical Formulation

**Quantization Function**:
```
Q_NF4(W) = quantize(W; dtype=nf4, blocksize=64)

Where NF4 encoding maps:
    W ∈ ℝ^(d×d) → W_q ∈ {-1, -0.6962, -0.5251, ..., 0.8924, 1.0} ^ (d×d)
    
Each block of 64 values shares:
    - Quantization constant c ∈ ℝ (stored in 8-bit)
    - Quantization offset (for asymmetric distributions)
```

**Dequantization for Forward Pass**:
```
W ≈ dequantize(W_q, c) = c · W_q
```

### Full Algorithm

```
Algorithm 13-QLoRA: Quantized Low-Rank Adaptation

Input: Pre-trained model θ₀ = {Wq, Wk, Wv, Wo, ...} in FP16/BF16
       Low-rank r, quantization blocksize b = 64
       Training dataset D

# ========== PREPROCESSING (ONE-TIME) ==========
Quantize base model:
    for each weight matrix W ∈ θ₀ do
        # Step 1: 4-bit NF4 quantization
        W_q, c₁ = quantize_nf4(W, blocksize=b)
        
        # Step 2: Double quantization (quantize the constants)
        c₁_q, c₂ = quantize_fp8(c₁, blocksize=256)
        
        Store: W_q (4-bit), c₁_q (8-bit), c₂ (FP16)

# ========== ADAPTER INITIALIZATION ==========
Initialize LoRA adapters (high precision):
    For each adapted weight matrix W ∈ {Wq, Wv}:
        B ~ N(0, σ²) with shape (d, r) in BF16
        A ~ 0 with shape (r, d) in BF16
        α = 1

# ========== TRAINING LOOP ==========
for epoch = 1 to E do
    for each minibatch {(x, y)} ⊂ D of size B do
        # Forward pass
        h = Embedding(x)
        for layer l = 1 to L do
            # Dequantize base weights on-the-fly
            Wq_fp16^(l) = dequantize_nf4(Wq_q^(l), c₁_q^(l), c₂^(l))
            Wk_fp16^(l) = dequantize_nf4(Wk_q^(l), c₁_q^(l), c₂^(l))
            Wv_fp16^(l) = dequantize_nf4(Wv_q^(l), c₁_q^(l), c₂^(l))
            Wo_fp16^(l) = dequantize_nf4(Wo_q^(l), c₁_q^(l), c₂^(l))
            
            # Compute with LoRA injection
            Q = h · (Wq_fp16^(l) + α · B_q^(l) · A_q^(l))  # QLoRA
            K = h · Wk_fp16^(l)                             # Frozen (quantized)
            V = h · (Wv_fp16^(l) + α · B_v^(l) · A_v^(l))  # QLoRA
            
            A = softmax((Q · K^T) / √d_k)
            h_attn = A · V · Wo_fp16^(l)
            h_mlp = MLP(h_attn; dequantize_nf4(W_mlp_q^(l)))
            h = h_attn + h_mlp
        
        ŷ = h · W_out
        L = CrossEntropy(ŷ, y)
        
        # Backward pass (only update B, A; base weights frozen)
        ∇B, ∇A = ∂L/∂B, ∂L/∂A
        
        # Optimizer step with paged memory management
        with unified_memory():  # Handle gradient spikes
            B ← B - η · ∇B
            A ← A - η · ∇A
        
Output: Low-rank adapters {B, A} (base model remains quantized)
```

---

## Memory Analysis

### QLoRA Memory Breakdown

For a model with `p` total parameters, adapting `k` layers with rank `r`:

```
Base model (4-bit NF4):
    W_q:  p × 0.5 bytes           (4-bit quantized weights)
    c₁_q: (p/64) × 1 byte         (8-bit quantization constants)
    c₂:   (p/64/256) × 2 bytes    (FP16 double-quant constants)
    ≈ 0.52p bytes

LoRA adapters (BF16):
    B, A: 2 × k × r × d × 2 bytes  (high-precision adapters)

Gradients (BF16):
    ∇B, ∇A: 2 × k × r × d × 2 bytes

Optimizer states (Adam):
    m, v: 2 × (2 × k × r × d) × 4 bytes  (FP32 momentum/variance)

Total: ~0.52p + 4krd + 4krd + 16krd ≈ 0.52p + 24krd bytes
```

### Comparison Table

| Method | Memory Formula | LLaMA 65B Example |
|--------|---------------|-------------------|
| **Full Fine-tuning (FP16)** | 12p | ~780 GB |
| **LoRA (FP16 base)** | 2p + 24krd | ~156 GB |
| **QLoRA (4-bit base)** | 0.52p + 24krd | ~48 GB |

**Conclusion**: QLoRA achieves **~16× memory reduction** vs full fine-tuning, enabling 65B model fine-tuning on a **single 48GB GPU**.

---

## Theoretical Justification

### Why 4-bit Quantization Works

1. **Weight Distribution**: Pre-trained transformer weights follow approximately normal distributions
2. **NF4 Optimality**: For normally distributed data, NF4 minimizes information loss under 4-bit constraint
3. **Gradient Flow**: Adapters (B, A) receive full-precision gradients; base model never updated
4. **Capacity Preservation**: Low-rank adaptation focuses updates on task-specific directions

### When QLoRA May Fail

Based on theoretical analysis, QLoRA is expected to degrade when:

1. **Extreme Quantization Sensitivity**: Layers with high variance or non-normal weight distributions
2. **Insufficient Rank**: r too low to capture task complexity
3. **Weight Similarity Breakdown**: If `cos_sim(W_LoRA, W_QLoRA) < 0.95`, significant information loss occurs

---

## Implementation Notes

### Libraries
- **bitsandbytes**: Provides NF4 quantization primitives
- **PEFT (Parameter-Efficient Fine-Tuning)**: HuggingFace library for LoRA/QLoRA
- **unsloth**: Optimized implementation with fused kernels

### Practical Considerations
1. **Blocksize**: 64 is optimal for NF4 (balances precision vs memory)
2. **Double Quantization**: Saves ~0.4 bytes per parameter (critical at scale)
3. **Compute Dtype**: Use BF16 for adapters (better than FP16 for training stability)
4. **Gradient Checkpointing**: Essential for fitting longer sequences in memory

---

## References

1. Phuong, M., & Hutter, M. (2022). Formal Algorithms for Transformers. arXiv:2207.09238
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022
3. Dettmers, T., et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023

---

**This extension demonstrates how QLoRA builds upon the transformer training foundations (Algorithm 13) by strategically combining quantization (memory reduction) with low-rank adaptation (parameter efficiency) while preserving gradient flow through high-precision adapters.**