"""
Test fine-tuned model vs base model

Usage:
    python test.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# ===========================================
# Configuration
# ===========================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./qwen_lora_adapter"


# ===========================================
# Setup
# ===========================================

def get_device():
    if torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32

DEVICE, DTYPE = get_device()
print(f"Using device: {DEVICE}")


# ===========================================
# Load Models
# ===========================================

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True
).to(DEVICE).eval()

print("Loading fine-tuned model...")
ft_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, trust_remote_code=True
)
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_PATH).to(DEVICE).eval()

print("Models ready!\n")


# ===========================================
# Generation
# ===========================================

def generate(model, question):
    """Generate response from model."""
    system = (
        "You are a compassionate emotional support assistant. "
        "Give warm, insightful responses about human emotions. "
        "Keep answers concise (2-3 sentences) and end with a follow-up question."
    )
    
    prompt = f"""<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    text = tokenizer.decode(output[0], skip_special_tokens=False)
    
    # Extract assistant response
    if "<|im_start|>assistant" in text:
        response = text.split("<|im_start|>assistant")[-1]
        response = response.split("<|im_end|>")[0].strip()
        return response
    return text[len(prompt):]


# ===========================================
# Interactive Chat
# ===========================================

def main():
    print("=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print("Type a question about emotions.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
        
        print("\n" + "-" * 50)
        
        print("\n[BASE MODEL]")
        print(generate(base_model, question))
        
        print("\n[FINE-TUNED]")
        print(generate(ft_model, question))
        
        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
