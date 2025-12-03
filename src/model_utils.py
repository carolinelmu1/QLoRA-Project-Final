"""
Model utilities for LoRA and QLoRA configuration
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import gc


def get_model_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def clear_memory():
    """Clear GPU and system memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model_16bit(model_name="gpt2-medium", device="cuda"):
    """
    Load base model in 16-bit precision (standard LoRA baseline)
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on
    
    Returns:
        model, tokenizer
    """
    print(f"Loading {model_name} in 16-bit precision...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    mem_usage = get_model_memory_usage()
    print(f"✓ Model loaded. Memory usage: {mem_usage:.2f} MB")
    
    return model, tokenizer


def load_base_model_4bit(model_name="gpt2-medium"):
    """
    Load base model with 4-bit NF4 quantization (QLoRA)
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        model, tokenizer
    """
    print(f"Loading {model_name} with 4-bit NF4 quantization...")
    
    # Configure 4-bit quantization with NF4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in BF16
        bnb_4bit_use_double_quant=True,      # Double quantization
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training (gradient checkpointing, etc.)
    model = prepare_model_for_kbit_training(model)
    
    mem_usage = get_model_memory_usage()
    print(f"✓ Model loaded and quantized. Memory usage: {mem_usage:.2f} MB")
    
    return model, tokenizer


def setup_lora_16bit(model, rank=8, target_modules=None, lora_alpha=16, lora_dropout=0.05):
    """
    Setup standard LoRA (16-bit base model)
    
    Args:
        model: Base model
        rank: LoRA rank (r)
        target_modules: Which modules to adapt (default: query and value projections)
        lora_alpha: Scaling parameter
        lora_dropout: Dropout for LoRA layers
    
    Returns:
        PEFT model with LoRA adapters
    """
    if target_modules is None:
        # For GPT-2: adapt attention query and value projections
        target_modules = ["c_attn"]  # GPT-2 uses c_attn for Q,K,V combined
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ LoRA configured:")
    print(f"  Rank: {rank}")
    print(f"  Target modules: {target_modules}")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def setup_lora_4bit(model, rank=8, target_modules=None, lora_alpha=16, lora_dropout=0.05):
    """
    Setup QLoRA (4-bit quantized base + LoRA adapters)
    
    Args:
        model: Quantized base model (from load_base_model_4bit)
        rank: LoRA rank (r)
        target_modules: Which modules to adapt
        lora_alpha: Scaling parameter
        lora_dropout: Dropout for LoRA layers
    
    Returns:
        PEFT model with QLoRA configuration
    """
    if target_modules is None:
        target_modules = ["c_attn"]
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"✓ QLoRA configured:")
    print(f"  Rank: {rank}")
    print(f"  Target modules: {target_modules}")
    print(f"  Base model: 4-bit NF4 quantized")
    print(f"  Adapters: BF16 precision")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model


def get_target_modules_by_layer_type(layer_type="qv"):
    """
    Get target modules for different layer adaptation strategies
    
    Args:
        layer_type: One of:
            - "qv": Query and Value (default)
            - "kv": Key and Value  
            - "q": Query only
            - "v": Value only
            - "all_attn": All attention projections
            - "attn_mlp": Attention + MLP
    
    Returns:
        List of module names to target
    """
    # Note: GPT-2 architecture specifics
    # c_attn contains Q, K, V projections (would need custom handling to separate)
    # c_proj is the output projection
    # c_fc and c_proj in MLP
    
    target_map = {
        "qv": ["c_attn"],  # Default: adapt combined QKV projection
        "kv": ["c_attn"],  # GPT-2 doesn't separate these easily
        "q": ["c_attn"],
        "v": ["c_attn"],
        "all_attn": ["c_attn", "c_proj"],  # Attention input + output
        "attn_mlp": ["c_attn", "c_proj", "c_fc"],  # + MLP
    }
    
    return target_map.get(layer_type, ["c_attn"])


def print_model_architecture(model):
    """Print trainable vs frozen parameters by layer"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"✓ TRAINABLE: {name:50s} {param.numel():>12,}")
        else:
            print(f"  FROZEN:    {name:50s} {param.numel():>12,}")
    
    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")