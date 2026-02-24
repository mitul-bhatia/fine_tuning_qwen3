# Fine-Tuning Qwen 2.5 1.5B with LoRA

Fine-tuning a small language model for humanized emotional conversations using the KnowSLM paper methodology.

## Assignment

> Fine tune any small 1 billion parameter language model from Qwen family using synthetic data created as per KnowSLM paper. Dataset to be prepared synthetically using ChatGPT of 100 input, output pairs for fine tuning.

## Results

| Metric | Value |
|--------|-------|
| Model | Qwen/Qwen2.5-1.5B-Instruct |
| Dataset | 100 synthetic Q&A pairs |
| Training Time | ~82 seconds |
| Final Loss | 0.96 |
| Token Accuracy | 78.6% |
| Trainable Parameters | 4.3M (0.28%) |

## Project Structure

```
├── train.py                 # Training script
├── test.py                  # Interactive testing script
├── emotions_dataset.csv     # 100 Q&A training pairs
├── qwen_lora_adapter/       # Trained LoRA adapter weights
└── report.pdf               # Detailed report
```

## Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets peft trl accelerate pandas
```

## Training

```bash
python train.py
```

## Testing

```bash
python test.py
```

Compare base model vs fine-tuned model responses interactively.

## Example Output

**Question:** "Why do I feel anxious even when everything is going well?"

**Base Model:**
> It's understandable to feel anxious sometimes! Here are some steps you can take to manage your anxiety: 1. Deep Breathing... 2. Mindfulness... 3. Physical Activity...

**Fine-Tuned:**
> Anxiety can appear even in good times because the brain sometimes scans for potential threats regardless of current circumstances. It's a protective mechanism that doesn't always match reality. What usually triggers that feeling for you?

## LoRA Configuration

- Rank: 16
- Alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj
- Dropout: 0.1

## Reference

- KnowSLM Paper: [arXiv:2504.04569](https://arxiv.org/abs/2504.04569)
