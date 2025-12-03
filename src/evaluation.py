"""
Evaluation utilities for comparing LoRA and QLoRA models
"""

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text from a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_token_match(model1, model2, tokenizer, test_prompts, max_length=50):
    """
    Compare token-level outputs between two models
    
    Args:
        model1: First model (e.g., LoRA)
        model2: Second model (e.g., QLoRA)
        tokenizer: Shared tokenizer
        test_prompts: List of prompts to test
        max_length: Generation length
    
    Returns:
        Dictionary with token match statistics
    """
    print(f"Evaluating token match on {len(test_prompts)} prompts...")
    
    match_scores = []
    
    for prompt in tqdm(test_prompts, desc="Token matching"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model1.device)
        
        with torch.no_grad():
            # Generate from both models
            out1 = model1.generate(
                **inputs,
                max_length=max_length,
                do_sample=False,  # Greedy for deterministic comparison
                pad_token_id=tokenizer.eos_token_id
            )
            
            out2 = model2.generate(
                **inputs,
                max_length=max_length,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Compare tokens
        min_len = min(out1.shape[1], out2.shape[1])
        matches = (out1[0, :min_len] == out2[0, :min_len]).sum().item()
        match_rate = matches / min_len
        match_scores.append(match_rate)
    
    return {
        "mean_token_match": np.mean(match_scores),
        "std_token_match": np.std(match_scores),
        "min_token_match": np.min(match_scores),
        "max_token_match": np.max(match_scores),
    }


def evaluate_embedding_similarity(model1, model2, tokenizer, test_prompts, max_length=50):
    """
    Compare output embeddings between two models using cosine similarity
    
    Args:
        model1: First model
        model2: Second model
        tokenizer: Shared tokenizer
        test_prompts: List of prompts
        max_length: Generation length
    
    Returns:
        Dictionary with cosine similarity statistics
    """
    print(f"Evaluating embedding similarity on {len(test_prompts)} prompts...")
    
    cosine_scores = []
    
    for prompt in tqdm(test_prompts, desc="Embedding similarity"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model1.device)
        
        with torch.no_grad():
            # Get final hidden states from both models
            out1 = model1(**inputs, output_hidden_states=True)
            out2 = model2(**inputs, output_hidden_states=True)
            
            # Use last hidden state, average over sequence
            emb1 = out1.hidden_states[-1].mean(dim=1).cpu().numpy()  # (1, hidden_dim)
            emb2 = out2.hidden_states[-1].mean(dim=1).cpu().numpy()
            
            # Cosine similarity
            cos_sim = cosine_similarity(emb1, emb2)[0, 0]
            cosine_scores.append(cos_sim)
    
    return {
        "mean_cosine_similarity": np.mean(cosine_scores),
        "std_cosine_similarity": np.std(cosine_scores),
        "min_cosine_similarity": np.min(cosine_scores),
        "max_cosine_similarity": np.max(cosine_scores),
    }


def compare_weight_matrices(model_lora, model_qlora, layer_name="transformer.h.0.attn.c_attn"):
    """
    Compare weight matrices between LoRA and QLoRA models
    
    Args:
        model_lora: LoRA model (16-bit)
        model_qlora: QLoRA model (4-bit)
        layer_name: Name of layer to compare
    
    Returns:
        Dictionary with weight similarity metrics
    """
    print(f"Comparing weight matrices for layer: {layer_name}")
    
    # Extract weights (this requires accessing the merged weights)
    # For PEFT models, we need to get the effective weight: W + B*A
    
    try:
        # Get LoRA weights
        lora_weight = None
        qlora_weight = None
        
        for name, param in model_lora.named_parameters():
            if layer_name in name and "lora_A" in name:
                lora_A = param.detach().cpu().numpy()
            if layer_name in name and "lora_B" in name:
                lora_B = param.detach().cpu().numpy()
        
        for name, param in model_qlora.named_parameters():
            if layer_name in name and "lora_A" in name:
                qlora_A = param.detach().cpu().numpy()
            if layer_name in name and "lora_B" in name:
                qlora_B = param.detach().cpu().numpy()
        
        # Compute effective LoRA updates: ΔW = B @ A
        lora_delta = lora_B @ lora_A
        qlora_delta = qlora_B @ qlora_A
        
        # Flatten for comparison
        lora_flat = lora_delta.flatten()
        qlora_flat = qlora_delta.flatten()
        
        # Compute cosine similarity
        cos_sim = cosine_similarity(
            lora_flat.reshape(1, -1),
            qlora_flat.reshape(1, -1)
        )[0, 0]
        
        # Compute L2 distance
        l2_dist = np.linalg.norm(lora_flat - qlora_flat)
        
        # Compute relative difference
        rel_diff = l2_dist / (np.linalg.norm(lora_flat) + 1e-8)
        
        return {
            "cosine_similarity": cos_sim,
            "l2_distance": l2_dist,
            "relative_difference": rel_diff,
            "lora_norm": np.linalg.norm(lora_flat),
            "qlora_norm": np.linalg.norm(qlora_flat),
        }
    
    except Exception as e:
        print(f"Error comparing weights: {e}")
        return None


def evaluate_instruction_following(model, tokenizer, test_instructions, max_length=100):
    """
    Qualitative evaluation of instruction-following capability
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_instructions: List of test instructions
        max_length: Max generation length
    
    Returns:
        List of (instruction, response) tuples
    """
    print(f"Generating responses for {len(test_instructions)} instructions...")
    
    results = []
    
    for instruction in tqdm(test_instructions, desc="Generating"):
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        response = generate_text(model, tokenizer, prompt, max_length=max_length)
        results.append((instruction, response))
    
    return results


def create_test_prompts(dataset, num_prompts=50):
    """Create test prompts from dataset"""
    # Sample random instructions
    indices = np.random.choice(len(dataset), size=min(num_prompts, len(dataset)), replace=False)
    
    prompts = []
    for idx in indices:
        example = dataset[int(idx)]
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        prompts.append(prompt)
    
    return prompts


def comprehensive_evaluation(
    model_lora,
    model_qlora,
    tokenizer,
    eval_dataset,
    num_test_prompts=50
):
    """
    Run comprehensive evaluation comparing LoRA and QLoRA
    
    Args:
        model_lora: LoRA model
        model_qlora: QLoRA model
        tokenizer: Tokenizer
        eval_dataset: Evaluation dataset
        num_test_prompts: Number of prompts to test
    
    Returns:
        DataFrame with all evaluation metrics
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION: LoRA vs QLoRA")
    print("="*60 + "\n")
    
    # Create test prompts
    test_prompts = create_test_prompts(eval_dataset, num_prompts=num_test_prompts)
    
    # Token match evaluation
    token_metrics = evaluate_token_match(model_lora, model_qlora, tokenizer, test_prompts)
    
    # Embedding similarity evaluation
    embedding_metrics = evaluate_embedding_similarity(
        model_lora, model_qlora, tokenizer, test_prompts
    )
    
    # Weight comparison (for selected layers)
    weight_metrics = compare_weight_matrices(model_lora, model_qlora)
    
    # Compile results
    results = {
        **token_metrics,
        **embedding_metrics,
    }
    
    if weight_metrics:
        results.update({f"weight_{k}": v for k, v in weight_metrics.items()})
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:40s}: {value:.4f}")
        else:
            print(f"{key:40s}: {value}")
    print("="*60 + "\n")
    
    return results


def analyze_layer_sensitivity(models_dict, layer_names, eval_dataset, tokenizer):
    """
    Analyze which layers are most sensitive to quantization
    
    Args:
        models_dict: Dictionary of {config_name: model}
        layer_names: List of layer names to analyze
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
    
    Returns:
        DataFrame with layer sensitivity results
    """
    print("\n" + "="*60)
    print("LAYER SENSITIVITY ANALYSIS")
    print("="*60 + "\n")
    
    results = []
    
    # TODO: Implement layer-by-layer comparison
    # This requires training models with different layer configurations
    
    return pd.DataFrame(results)