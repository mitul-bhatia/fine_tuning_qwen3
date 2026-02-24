# Fine-Tuning Qwen 2.5 1.5B with LoRA for Emotional Support Conversations

**Assignment Report**  
**Course:** Open Elective - Large Language Models  
**Date:** February 24, 2026  
**Model:** Qwen/Qwen2.5-1.5B-Instruct  
**Methodology:** KnowSLM Paper (arXiv:2504.04569)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Assignment Requirements](#2-assignment-requirements)
3. [KnowSLM Paper Methodology](#3-knowslm-paper-methodology)
4. [Dataset Preparation](#4-dataset-preparation)
5. [Technical Implementation](#5-technical-implementation)
6. [Training Process](#6-training-process)
7. [Results and Analysis](#7-results-and-analysis)
8. [Code Documentation](#8-code-documentation)
9. [Usage Instructions](#9-usage-instructions)
10. [Conclusion](#10-conclusion)

---

## 1. Executive Summary

This project demonstrates fine-tuning a small language model (Qwen 2.5 1.5B) using **Low-Rank Adaptation (LoRA)** on a synthetically generated emotional conversation dataset. The methodology follows the **KnowSLM paper** guidelines for creating humanized, knowledge-augmented conversational AI.

### Key Achievements

| Metric | Value |
|--------|-------|
| Model | Qwen2.5-1.5B-Instruct |
| Dataset Size | 100 conversation pairs |
| Training Time | ~82 seconds |
| Final Loss | 0.96 |
| Token Accuracy | 78.6% |
| Trainable Parameters | 4.3M (0.28% of total) |

---

## 2. Assignment Requirements

The assignment specified:

> *Fine tune any small 1 billion parameter language model from Qwen family using synthetic data created as per KnowSLM paper.*

### Checklist

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| Qwen family model (~1B params) | ✅ Complete | Qwen2.5-1.5B-Instruct |
| Synthetic dataset | ✅ Complete | 100 Q&A pairs on emotions |
| KnowSLM methodology | ✅ Complete | Follow-up questions, 2-3 sentence responses |
| 100 input/output pairs | ✅ Complete | emotions_dataset.csv |
| Fine-tuning complete | ✅ Complete | LoRA adapter saved |

---

## 3. KnowSLM Paper Methodology

### Paper Overview

**Title:** KnowSLM: A framework for evaluation of small language models for knowledge augmentation and humanised conversations

**Key Findings from the Paper:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    KnowSLM FRAMEWORK                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SYNTHETIC DATA GENERATION                                   │
│     └── Use LLM (ChatGPT) to create conversation pairs          │
│     └── Questions: varied openers (why, when, how, what...)     │
│     └── Answers: 2-3 sentences + follow-up question             │
│                                                                 │
│  2. FINE-TUNING WITH LoRA                                       │
│     └── Low-rank adaptation for efficiency                      │
│     └── Rank 8-16 recommended for balance                       │
│     └── Target attention layers (q_proj, v_proj, etc.)          │
│                                                                 │
│  3. EVALUATION                                                  │
│     └── Compare base vs fine-tuned responses                    │
│     └── LLM-judge for quality assessment                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Data Generation Prompts (from KnowSLM)

**Question Generation:**
```
Generate one conversation initiating statement in English.
Frame the conversation in a natural tone and use different
starting formats of questioning like 'why', 'when', 'where'.
Ensure the questions feel engaging and unique.
```

**Answer Generation:**
```
Give an informative response in English in 2 lines.
Ask a thoughtful question at the end.
Don't generate further question and answer statements.
```

### Key Paper Insights Applied

| Paper Finding | Our Implementation |
|---------------|-------------------|
| "Rank 8 balances efficiency and expressiveness" | Used rank=16 for better adaptation |
| "Fine-tuning better for tone/style" | Focus on empathetic emotional style |
| "Small datasets (<100k tokens) show minimal improvement" | Used 100 high-quality examples |
| "Follow-up questions improve conversation flow" | Every response ends with a question |

---

## 4. Dataset Preparation

### Dataset Structure

**File:** `emotions_dataset.csv`  
**Format:** CSV with two columns

```csv
input,output
"Why do people cry even when they're happy?","Happy tears often show up when emotions overflow and the body needs a release valve. It's the nervous system balancing intense joy with physical expression so you can process the moment. Have you ever noticed when that happens most for you?"
```

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Examples | 100 |
| Topics Covered | 20+ emotional themes |
| Avg Input Length | ~8-12 words |
| Avg Output Length | ~40-60 words |
| Response Format | 2-3 sentences + follow-up |

### Topics Covered

```
┌──────────────────┬──────────────────┬──────────────────┐
│    EMOTIONS      │    BEHAVIORS     │    PSYCHOLOGY    │
├──────────────────┼──────────────────┼──────────────────┤
│ • Happiness      │ • Procrastination│ • Self-doubt     │
│ • Sadness        │ • People-pleasing│ • Impostor       │
│ • Anxiety        │ • Overthinking   │   syndrome       │
│ • Anger          │ • Avoidance      │ • Perfectionism  │
│ • Jealousy       │ • Rumination     │ • Attachment     │
│ • Grief          │ • Emotional      │ • Trust issues   │
│ • Loneliness     │   suppression    │ • Fear of        │
│ • Fear           │ • Boundary       │   rejection      │
│ • Shame          │   setting        │ • Resilience     │
│ • Nostalgia      │                  │                  │
└──────────────────┴──────────────────┴──────────────────┘
```

### Sample Data Points

| # | Input | Output |
|---|-------|--------|
| 1 | "Why do people cry even when they're happy?" | "Happy tears often show up when emotions overflow and the body needs a release valve. It's the nervous system balancing intense joy with physical expression. Have you ever noticed when that happens most for you?" |
| 2 | "How does anxiety grow out of small worries?" | "Anxiety tends to build when the mind keeps rehearsing worst-case scenarios and never gets reassurance. Each repeated thought adds a layer until it feels overwhelming. What kind of worries tend to loop most for you?" |
| 3 | "What makes someone procrastinate even when they care?" | "Procrastination is often about avoiding discomfort, not laziness, especially when fear of failure is involved. The brain chooses short-term relief over long-term goals. Do you notice what feelings come up right before you delay something?" |

---

## 5. Technical Implementation

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. DATA LOADING                                                │
│     emotions_dataset.csv → pandas DataFrame → HuggingFace Dataset│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. CHATML FORMATTING                                           │
│     <|im_start|>system                                          │
│     {system_prompt}<|im_end|>                                   │
│     <|im_start|>user                                            │
│     {input}<|im_end|>                                           │
│     <|im_start|>assistant                                       │
│     {output}<|im_end|>                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. MODEL LOADING                                               │
│     Qwen2.5-1.5B-Instruct (1.54 billion parameters)            │
│     Device: MPS (Apple Silicon GPU)                             │
│     Precision: float16                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. LoRA ADAPTER INJECTION                                      │
│     Target: q_proj, k_proj, v_proj, o_proj                      │
│     Trainable: 4.3M params (0.28%)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. TRAINING (SFTTrainer)                                       │
│     2 epochs, batch_size=1, gradient_accumulation=4             │
│     Learning rate: 2e-4 with cosine schedule                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. SAVE ADAPTER                                                │
│     ./qwen_lora_adapter/ (~17 MB)                               │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA Configuration

**What is LoRA?**

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning technique that:
- Freezes the base model weights
- Adds small trainable matrices to specific layers
- Reduces memory requirements by ~10x

```
┌─────────────────────────────────────────────────────────────────┐
│                    LoRA MATHEMATICS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Original weight matrix W: [1536 × 1536] = 2.36M parameters     │
│                                                                 │
│  LoRA decomposition:                                            │
│  ┌────────┐     ┌────────┐                                      │
│  │   A    │  ×  │   B    │  =  ΔW                              │
│  │1536×16 │     │ 16×1536│                                      │
│  └────────┘     └────────┘                                      │
│   24,576    +    24,576   =  49,152 parameters                  │
│                                                                 │
│  Reduction: 2.36M → 49K (48x smaller!)                          │
│                                                                 │
│  Final output: W + (α/r) × A × B                                │
│  Where α=32, r=16, so scaling = 2.0                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Our LoRA Settings:**

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `r` (rank) | 16 | Compression level - higher = more expressive |
| `lora_alpha` | 32 | Scaling factor (alpha/r = 2.0) |
| `lora_dropout` | 0.1 | Regularization to prevent overfitting |
| `target_modules` | q_proj, k_proj, v_proj, o_proj | All attention projections |
| `bias` | none | Don't train bias terms |

### Why These Target Modules?

```
┌─────────────────────────────────────────────────────────────────┐
│                 TRANSFORMER ATTENTION LAYER                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input → [q_proj] → Query  ─┐                                   │
│       → [k_proj] → Key    ──┼──→ Attention ──→ [o_proj] → Output│
│       → [v_proj] → Value  ──┘                                   │
│                                                                 │
│  LoRA adapters on all 4 projections:                            │
│  • q_proj: "What to look for" - learns new query patterns      │
│  • k_proj: "What info exists" - improves context matching      │
│  • v_proj: "What to output" - controls response content        │
│  • o_proj: "How to combine" - refines final representation     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Training Process

### Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 1 | Memory constraint on Mac |
| Gradient Accumulation | 4 | Effective batch = 4 |
| Learning Rate | 2e-4 | Standard for LoRA |
| Epochs | 2 | Prevent overfitting on small data |
| Max Sequence Length | 256 | Sufficient for our Q&A format |
| Optimizer | AdamW | Standard choice |
| Scheduler | Cosine | Smooth learning rate decay |
| Precision | FP16 | Memory efficient on MPS |

### Training Progress

```
Epoch 0.4  │ Loss: 3.42  │ Accuracy: 42.6%  │ ████░░░░░░░░░░
Epoch 0.8  │ Loss: 2.06  │ Accuracy: 59.1%  │ ████████░░░░░░
Epoch 1.2  │ Loss: 1.18  │ Accuracy: 75.2%  │ ██████████░░░░
Epoch 1.6  │ Loss: 1.03  │ Accuracy: 76.7%  │ ████████████░░
Epoch 2.0  │ Loss: 0.96  │ Accuracy: 78.6%  │ ██████████████
```

### Loss Curve Visualization

```
Loss
  │
4 ┤ ●
  │  ╲
3 ┤   ╲
  │    ╲
2 ┤     ●
  │      ╲
1 ┤       ●━━●━━●
  │
0 ┼────────────────────
  0    0.5    1    1.5    2  Epochs
```

### Training Metrics Summary

| Metric | Start | End | Improvement |
|--------|-------|-----|-------------|
| Loss | 3.42 | 0.96 | ↓ 72% |
| Token Accuracy | 42.6% | 78.6% | ↑ 36 points |
| Learning Rate | 2e-4 | 5e-6 | Cosine decay |

---

## 7. Results and Analysis

### Model Comparison: Base vs Fine-Tuned

**Test Question:** *"Why do I feel anxious even when everything is going well?"*

#### Base Model Response:
```
It's understandable to feel anxious sometimes! Here are some steps you 
can take to manage your anxiety:

1. **Deep Breathing**: Practice deep breathing exercises. Inhale slowly 
   through your nose, hold for a few seconds, and exhale slowly through 
   your mouth.

2. **Mindfulness or Meditation**: Engage in mindfulness meditation where 
   you focus on the present moment without judgment.

3. **Physical Activity**: Exercise is great for reducing stress and 
   improving mood. Even a short walk can make a difference.

4. **Connect with Others**: Talk about how you're feeling with someone 
   you trust.

5. **Routine**: Try maintaining a daily routine if possible.
```

#### Fine-Tuned Model Response:
```
Anxiety can appear even in good times because the brain sometimes scans 
for potential threats regardless of current circumstances. It's a 
protective mechanism that doesn't always match reality. What usually 
triggers that feeling for you?
```

### Comparison Analysis

| Aspect | Base Model | Fine-Tuned |
|--------|------------|------------|
| Length | Long (5 paragraphs) | Concise (3 sentences) |
| Style | List-based, clinical | Conversational, warm |
| Follow-up | None | "What triggers that for you?" |
| Tone | Instructional | Empathetic |
| KnowSLM Aligned | ❌ | ✅ |

### What Fine-Tuning Changed

```
┌─────────────────────────────────────────────────────────────────┐
│                    STYLE TRANSFORMATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BEFORE (Base Model):                                           │
│  • Generic advice format                                        │
│  • Numbered lists                                               │
│  • Text-book explanation style                                  │
│  • No engagement with user                                      │
│                                                                 │
│  AFTER (Fine-Tuned):                                            │
│  • Personalized, warm tone                                      │
│  • 2-3 sentence responses                                       │
│  • Validates the emotion first                                  │
│  • Ends with follow-up question                                 │
│  • Feels like talking to a wise friend                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Code Documentation

### Project Structure

```
assignment_2_open_elective/
│
├── 2504.04569v2 (2).pdf      # KnowSLM research paper
│
├── emotions_dataset.csv       # 100 Q&A training pairs
│   └── Columns: input, output
│
├── train.py                   # Training script (~100 lines)
│   ├── Configuration section
│   ├── Data loading functions
│   ├── Model setup functions
│   └── Training function
│
├── test.py                    # Interactive testing script
│   ├── Model loading
│   ├── Generation function
│   └── Interactive chat loop
│
├── qwen_lora_adapter/         # Trained LoRA adapter
│   ├── adapter_config.json    # LoRA configuration
│   ├── adapter_model.safetensors  # Trained weights (~17 MB)
│   ├── tokenizer.json         # Tokenizer vocabulary
│   └── tokenizer_config.json  # Tokenizer settings
│
├── venv/                      # Python virtual environment
│
└── REPORT.md                  # This report
```

### train.py Overview

```python
# Configuration
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_RANK = 16
LORA_ALPHA = 32
EPOCHS = 2

# Pipeline
def main():
    dataset = load_dataset("emotions_dataset.csv")  # Load & format data
    model, tokenizer = load_model()                  # Load Qwen
    model = apply_lora(model)                        # Add LoRA adapters
    train(model, tokenizer, dataset)                 # Run SFTTrainer
```

### test.py Overview

```python
# Load both models
base_model = load_base_model()
ft_model = load_finetuned_model()

# Interactive loop
while True:
    question = input("Question: ")
    print("[BASE]", generate(base_model, question))
    print("[FINE-TUNED]", generate(ft_model, question))
```

---

## 9. Usage Instructions

### Prerequisites

- Python 3.10+
- macOS with Apple Silicon (MPS) or Linux/Windows with CUDA
- ~8 GB RAM minimum

### Setup

```bash
# Navigate to project
cd /path/to/assignment_2_open_elective

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets peft trl accelerate pandas
```

### Training (if needed)

```bash
# Run training script
python train.py

# Output:
# - Loads dataset (100 examples)
# - Downloads Qwen model (~3 GB first time)
# - Applies LoRA adapters
# - Trains for 2 epochs (~80 seconds)
# - Saves adapter to ./qwen_lora_adapter/
```

### Testing

```bash
# Run interactive test
python test.py

# Example interaction:
# Question: Why do I feel lonely even with friends?
# [BASE MODEL] ... (long response)
# [FINE-TUNED] ... (concise, empathetic response with follow-up)
```

### Using the Adapter in Your Code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base, "./qwen_lora_adapter")

# Generate response
prompt = "<|im_start|>system\nYou are compassionate...<|im_end|>\n<|im_start|>user\nWhy do I overthink?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

---

## 10. Conclusion

### What We Accomplished

1. **Successfully fine-tuned** Qwen 2.5 1.5B using LoRA
2. **Created synthetic dataset** following KnowSLM methodology
3. **Achieved significant style adaptation** from generic to empathetic
4. **Reduced training overhead** to 0.28% of parameters
5. **Documented the entire process** for reproducibility

### Key Learnings

| Topic | Insight |
|-------|---------|
| LoRA | Enables fine-tuning on consumer hardware |
| Dataset Quality | Format and style matter more than quantity |
| KnowSLM | Follow-up questions improve conversation flow |
| Evaluation | Side-by-side comparison shows clear improvements |

### Limitations

- Small dataset (100 examples) limits generalization
- Some responses may still echo training data patterns
- Paper suggests 500+ examples for robust improvement

### Future Improvements

1. Expand dataset to 500+ examples
2. Add more diverse emotional topics
3. Implement LLM-judge evaluation
4. Try RAG augmentation for factual accuracy

---

## References

1. Harbola, C., & Purwar, A. (2025). *KnowSLM: A framework for evaluation of small language models for knowledge augmentation and humanised conversations.* arXiv:2504.04569

2. Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685

3. Qwen Team. (2024). *Qwen2.5 Technical Report.* Alibaba Cloud.

---

**Report Generated:** February 24, 2026  
**Total Training Time:** ~82 seconds  
**Final Model Location:** `./qwen_lora_adapter/`
