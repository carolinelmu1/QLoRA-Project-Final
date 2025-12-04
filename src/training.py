"""
Training utilities for LoRA and QLoRA experiments
"""

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import time
import pandas as pd
from typing import Dict, List
import numpy as np


def prepare_alpaca_dataset(tokenizer, max_length=512, num_samples=None):
    """
    Load and prepare Alpaca instruction-following dataset
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        num_samples: Number of samples to use (None = all)
    
    Returns:
        train_dataset, eval_dataset
    """
    print("Loading Alpaca dataset...")
    
    # Load Alpaca dataset from HuggingFace
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    if num_samples:
        dataset = dataset.select(range(num_samples))
        print(f"Using {num_samples} samples for faster training")
    
    # Format prompts
    def format_instruction(example):
        """Format Alpaca instruction-input-output into a single prompt"""
        instruction = example["instruction"]
        input_text = example["input"]
        output_text = example["output"]
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
        
        return {"text": prompt}
    
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Split train/eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"✓ Dataset prepared:")
    print(f"  Training samples: {len(split_dataset['train'])}")
    print(f"  Evaluation samples: {len(split_dataset['test'])}")
    
    return split_dataset["train"], split_dataset["test"]


class MemoryTrackingTrainer(Trainer):
    """Custom trainer that tracks memory usage during training"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_log = []
        self.peak_memory = 0
        
    def training_step(self, model, inputs, *args):
        """Override training step to track memory"""
        
        # Memory before forward pass
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # Standard training step
        loss = super().training_step(model, inputs, *args)
        
        # Track peak memory
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, peak_mem)
            self.memory_log.append(peak_mem)
        
        return loss
    
    def get_memory_stats(self):
        """Return memory statistics"""
        if not self.memory_log:
            return {}
        
        return {
            "peak_memory_mb": self.peak_memory,
            "mean_memory_mb": np.mean(self.memory_log),
            "std_memory_mb": np.std(self.memory_log)
        }


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir="./results",
    num_epochs=1,
    batch_size=4,
    learning_rate=2e-4,
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    max_steps=200,  # Limit for diagnostic experiments
):
    """
    Train model with memory tracking
    
    Args:
        model: PEFT model with LoRA/QLoRA
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory to save results
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        logging_steps: Log every N steps
        eval_steps: Evaluate every N steps
        save_steps: Save checkpoint every N steps
        max_steps: Maximum training steps (for quick experiments)
    
    Returns:
        trainer, training_results
    """
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        fp16=False,  # Use BF16 for QLoRA
        bf16=True,
        gradient_accumulation_steps=1,
        warmup_steps=10,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="none",  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM (not masked)
    )
    
    trainer = MemoryTrackingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print(f"\n{'='*60}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Max steps: {max_steps}")
    print(f"Epochs: {num_epochs}")
    print(f"{'='*60}\n")
    
    # Train
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    # Get memory stats
    memory_stats = trainer.get_memory_stats()
    
    # Compile results
    results = {
        "training_loss": train_result.training_loss,
        "training_time_seconds": training_time,
        "steps": train_result.global_step,
        "time_per_step": training_time / train_result.global_step if train_result.global_step > 0 else 0,
        **memory_stats
    }
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time: {training_time:.2f}s")
    print(f"Time per step: {results['time_per_step']:.3f}s")
    print(f"Peak memory: {memory_stats.get('peak_memory_mb', 0):.2f} MB")
    print(f"Final loss: {train_result.training_loss:.4f}")
    print(f"{'='*60}\n")
    
    return trainer, results


def run_experiment(
    model_name="gpt2-medium",
    quantization="16bit",  # "16bit" or "4bit"
    rank=8,
    target_modules=None,
    num_samples=1000,  # Small for diagnostic
    max_steps=200,
    batch_size=4,
    learning_rate=2e-4,
    output_dir="./results",
):
    """
    Run a complete experiment with specified configuration
    
    Args:
        model_name: Base model to use
        quantization: "16bit" (LoRA) or "4bit" (QLoRA)
        rank: LoRA rank
        target_modules: Modules to adapt
        num_samples: Number of training samples
        max_steps: Training steps
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Save directory
    
    Returns:
        Dictionary with experiment results
    """
    from model_utils import (
        load_base_model_16bit,
        load_base_model_4bit,
        setup_lora_16bit,
        setup_lora_4bit,
        clear_memory
    )
    
    clear_memory()
    
    experiment_name = f"{quantization}_r{rank}"
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: {experiment_name}")
    print(f"{'#'*70}\n")
    
    # Load model
    if quantization == "16bit":
        model, tokenizer = load_base_model_16bit(model_name)
        model = setup_lora_16bit(model, rank=rank, target_modules=target_modules)
    elif quantization == "4bit":
        model, tokenizer = load_base_model_4bit(model_name)
        model = setup_lora_4bit(model, rank=rank, target_modules=target_modules)
    else:
        raise ValueError(f"Unknown quantization: {quantization}")
    
    # Prepare data
    train_dataset, eval_dataset = prepare_alpaca_dataset(
        tokenizer,
        num_samples=num_samples
    )
    
    # Train
    trainer, results = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=f"{output_dir}/{experiment_name}",
        max_steps=max_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    # Save model
    model.save_pretrained(f"{output_dir}/{experiment_name}/final_model")
    
    # Add metadata
    results["experiment_name"] = experiment_name
    results["quantization"] = quantization
    results["rank"] = rank
    results["model_name"] = model_name
    results["num_samples"] = num_samples
    
    clear_memory()
    
    return results, model, tokenizer