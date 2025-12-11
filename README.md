# DSAN-5800-Final-Project
2025 Fall dsan 5800 final project
# LoRA Fine-Tuning for SST-2 Sentiment Classification

This repository implements parameter-efficient fine-tuning (PEFT) techniques for the SST-2 sentiment classification task, focusing on Low-Rank Adaptation (LoRA). We compare a frozen DistilBERT baseline with LoRA configurations of rank 8 and rank 16, and evaluate their efficiency and performance.

---

## 1. Project Overview

Fine-tuning large transformer models typically requires updating tens of millions of parameters, which is computationally expensive. Low-Rank Adaptation (LoRA) provides a more efficient alternative by inserting low-rank trainable matrices into attention layers while keeping the backbone frozen.

This project evaluates:

- Whether LoRA can outperform a frozen DistilBERT baseline  
- How LoRA performance changes with different ranks (8 vs. 16)  
- The relationship between parameter count and accuracy  

---

## 2. Dataset

The dataset used is SST-2 from the GLUE benchmark.

| Split      | Size  |
|------------|--------|
| Train      | 67,349 |
| Validation | 872    |
| Test       | 1,821  |

All text is tokenized using `distilbert-base-uncased` with maximum sequence length 128, padded and truncated appropriately, and formatted as PyTorch tensors.

---

## 3. Methods

### 3.1 Baseline: Frozen DistilBERT + Classification Head

- DistilBERT encoder is fully frozen.
- Only the classifier head is trainable.
- Trainable parameters: 592,130 (0.884% of full model).
- Trained for 3 epochs with HuggingFace Trainer.

This model serves as a minimal-effort baseline for comparison with LoRA.

### 3.2 LoRA Fine-Tuning

LoRA adds trainable low-rank matrices to the query (`q_lin`) and value (`v_lin`) projection layers in DistilBERT. Only the LoRA parameters are updated during training.

LoRA configurations evaluated:

| Rank | Trainable Parameters | Trainable % |
|------|----------------------|-------------|
| 8    | 739,586              | 1.0925%     |
| 16   | 887,042              | 1.3075%     |

Both LoRA models are trained for 3 epochs under the same conditions as the baseline.

---

## 4. Results

### 4.1 Validation Accuracy and Loss

| Model                        | Trainable % | Accuracy | Loss    |
|------------------------------|-------------|----------|---------|
| Baseline (frozen encoder)    | 0.884%      | 0.8337   | 0.3653  |
| LoRA (rank = 8)              | 1.0925%     | 0.8979   | 0.2911  |
| LoRA (rank = 16)             | 1.3075%     | 0.8911   | 0.2747  |

Observations:

- LoRA significantly outperforms the frozen baseline.
- LoRA rank 8 achieves the highest accuracy.
- Increasing rank raises parameter count but does not improve accuracy.

---

## 5. Error Analysis

Error patterns from the LoRA rank 8 model include:

1. Ambiguous or sarcastic sentiment  
2. Very short sentences lacking contextual cues  
3. Long or complex sentences where polarity appears late  
4. Neutral wording with unclear sentiment  

Short sentences show the highest misclassification rate, suggesting dataset ambiguity contributes more to errors than LoRA capacity.

---

## 6. Training Scripts

### Baseline Training  
File: `baseline_training.py`  
Includes preprocessing, model setup, freezing encoder, training, evaluation, and saving outputs to `baseline_results.json`.

### LoRA Training  
File: `lora_training.py`  
Implements LoRA injection, allows rank selection, trains for 3 epochs, and saves JSON result files such as:

- `lora_r8_results_*.json`
- `lora_r16_results_*.json`

### Visualization  
File: `plot_training_curves.py`  
Extracts checkpoint logs and produces training curves:

- `baseline_training_curves.png`

---

## 7. Repository Structure

├── baseline_training.py
├── lora_training.py
├── plot_training_curves.py
├── baseline_results.json
├── lora_r8_results_.json
├── lora_r16_results_.json
├── baseline_training_curves.png
├── README.md
└── report/
└── main.tex


---

## 8. Key Takeaways

- LoRA provides substantial accuracy improvements with minimal parameter updates.
- Rank 8 is the most efficient configuration for SST-2.
- Parameter-efficient fine-tuning is practical even with limited computational resources.
- Increasing LoRA rank beyond a certain point yields diminishing returns.

---
