"""
Fine-tune Qwen 2.5 1.5B on Emotional Conversations using LoRA
Following KnowSLM paper methodology

Requirements:
    pip install torch transformers datasets peft trl accelerate pandas
"""

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


# ===========================================
# Configuration
# ===========================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_PATH = "emotions_dataset.csv"
OUTPUT_DIR = "./qwen_lora_adapter"
MAX_LENGTH = 256

# LoRA settings (from KnowSLM paper recommendations)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training settings
BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
LEARNING_RATE = 2e-4
EPOCHS = 2


# ===========================================
# Device Setup
# ===========================================

def get_device():
    """Detect best available device for Mac."""
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU)")
        return "mps", torch.float16
    print("Using CPU")
    return "cpu", torch.float32

DEVICE, DTYPE = get_device()


# ===========================================
# Data Preparation
# ===========================================

def load_dataset(path):
    """Load CSV and format for training."""
    print(f"\nLoading dataset: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} examples")
    
    # System prompt for emotional support assistant
    system_prompt = (
        "You are a compassionate emotional support assistant. "
        "Give warm, insightful responses about human emotions and psychology. "
        "Keep answers concise (2-3 sentences) and end with a follow-up question."
    )
    
    # Format each row into ChatML format
    def format_row(row):
        text = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{row['input']}<|im_end|>
<|im_start|>assistant
{row['output']}<|im_end|>"""
        return {"text": text}
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: format_row(x))
    
    print(f"\nSample formatted text:\n{dataset[0]['text'][:300]}...")
    return dataset


# ===========================================
# Model Setup
# ===========================================

def load_model():
    """Load base model and tokenizer."""
    print(f"\nLoading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    
    print(f"Model parameters: {model.num_parameters():,}")
    return model, tokenizer


def apply_lora(model):
    """Apply LoRA adapters to model."""
    print(f"\nApplying LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})")
    
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    
    return model


# ===========================================
# Training
# ===========================================

def train(model, tokenizer, dataset):
    """Run training with SFTTrainer."""
    print("\nStarting training...")
    
    training_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        warmup_steps=10,
        logging_steps=10,
        save_strategy="epoch",
        fp16=(DEVICE == "mps"),
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        max_length=MAX_LENGTH,
        dataset_text_field="text",
        packing=False,
        dataloader_pin_memory=False,
    )
    
    model = model.to(DEVICE)
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_config,
    )
    
    trainer.train()
    
    # Save adapter
    print(f"\nSaving adapter to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return trainer


# ===========================================
# Main
# ===========================================

def main():
    print("=" * 50)
    print("QWEN FINE-TUNING WITH LORA")
    print("=" * 50)
    
    # Load data
    dataset = load_dataset(DATASET_PATH)
    
    # Load model
    model, tokenizer = load_model()
    
    # Apply LoRA
    model = apply_lora(model)
    
    # Train
    train(model, tokenizer, dataset)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
