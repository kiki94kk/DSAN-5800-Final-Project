# ANLY-5800 Final Project – Proposal

## 1. Team
* **Project title**: Efficient Sentiment Analysis via LoRA Fine-tuning
* **Team members**: [你的名字] (NetID: [你的NetID])
* **Preferred track**: (B) LoRA finetuning

---

## 2. Problem statement & motivation

### Task
Binary sentiment classification on movie reviews using parameter-efficient fine-tuning (PEFT) techniques, specifically Low-Rank Adaptation (LoRA).

### Why it matters
- **Scientific**: Full fine-tuning of large language models requires updating billions of parameters, which is computationally expensive and memory-intensive. LoRA demonstrates that task adaptation can be achieved by injecting trainable low-rank matrices into frozen models, reducing trainable parameters by >99% while maintaining competitive performance.
- **Practical**: PEFT techniques like LoRA enable resource-constrained practitioners to adapt large models on consumer hardware, democratizing access to state-of-the-art NLP capabilities.

### Desired outcome
By Week 3, we will:
1. Demonstrate that LoRA achieves ≥90% accuracy on SST-2 with <1% trainable parameters
2. Characterize the trade-off between LoRA rank and performance
3. Provide reproducible code and analysis of when LoRA is most effective

---

## 3. Datasets

### Primary dataset
**SST-2 (Stanford Sentiment Treebank - Binary)**
- **Source**: GLUE benchmark via Hugging Face `datasets` library
- **Size**: 
  - Training: 67,349 samples
  - Validation: 872 samples
  - Test: 1,821 samples
- **Task**: Binary classification (positive/negative movie reviews)
- **Domain**: Movie reviews in English

### Preprocessing
- Tokenization using pretrained tokenizer (DistilBERT or similar)
- Max sequence length: 128 tokens
- Padding and truncation applied
- No additional data augmentation initially

### Train/val/test split
Using standard GLUE splits:
- Train on 67K samples
- Validate/tune hyperparameters on 872 samples
- Report final metrics on test set (1.8K samples)

---

## 4. Baseline

### Baseline model/system
**Frozen backbone + linear classifier**:
- Pretrained DistilBERT-base-uncased (66M parameters)
- All transformer layers frozen
- Only train a linear classification head (768 → 2 classes)
- Trainable parameters: ~1,500 (0.002% of total)

### Baseline metrics
- **Primary**: Accuracy (standard for SST-2)
- **Secondary**: F1-score, training time, memory usage
- **Expected baseline**: 75-82% accuracy (pretrained features + simple classifier)

This baseline represents the minimum effort approach: leveraging pretrained representations without expensive fine-tuning.

---

## 5. Approach (beyond baseline)

### Core improvement: LoRA fine-tuning

**Configuration**:
- Inject low-rank adapters into attention projection layers (Q, V matrices)
- LoRA rank `r = 8` (primary), with ablation at `r = 16`
- Scaling factor `alpha = 16`
- Dropout = 0.1
- Expected trainable parameters: ~300K (0.45% of total)

**Hypothesis**: LoRA will allow task-specific adaptation of attention mechanisms, improving accuracy by 8-15% over the frozen baseline.

### Experiments planned

#### Experiment 1: Baseline vs. LoRA-8 (Week 2)
- **Question**: Does LoRA significantly outperform frozen baseline?
- **Setup**: Train LoRA with rank=8 on full training set
- **Metrics**: Accuracy, F1, parameter count, training time
- **Expected**: LoRA achieves ~90% accuracy vs. ~80% baseline

#### Experiment 2: Rank ablation study (Week 2-3)
- **Question**: How does LoRA rank affect performance and efficiency?
- **Setup**: Compare rank=8 vs. rank=16 (double capacity)
- **Analysis**: 
  - Parameter count vs. accuracy trade-off
  - Training curves (convergence speed)
  - Risk of overfitting with higher rank

#### Experiment 3: Error analysis (Week 3)
- **Qualitative**: Identify examples where LoRA succeeds/fails
- **Quantitative**: Break down errors by sentence length, sentiment strength
- **Insight**: When is LoRA most/least effective?

---

## 6. Compute & resources

### Jetstream2
**Yes**, will use JS2 for Week 2-3 experiments:
- **Why**: A100 GPU (40GB VRAM) enables faster training and larger batch sizes
- **Usage plan**:
  - Week 1: Local development (Mac/Colab for baseline)
  - Week 2-3: JS2 for LoRA experiments and ablations

### Resource estimates
- **Model size**: DistilBERT-base (66M params, ~250MB)
- **Batch size**: 16-32 per device
- **Training time**: 
  - Baseline: ~10 minutes (3 epochs on 67K samples)
  - LoRA: ~15-20 minutes per experiment
  - Total compute: <2 hours across all experiments

### Storage
- Datasets: ~100MB (cached SST-2)
- Model checkpoints: ~1GB (3 experiments × 3 epochs)
- Total: <5GB (well within 250GB limit)

---

## 7. Risks & scope

### Potential risks

**Risk 1: Baseline performs too well**
- If frozen baseline achieves >85% accuracy, LoRA's improvement may be marginal
- **Mitigation**: Still valuable to analyze parameter efficiency; consider harder dataset variant

**Risk 2: Hyperparameter sensitivity**
- LoRA performance may be sensitive to learning rate, rank choice
- **Mitigation**: Allocate time in Week 2 for hyperparameter search

**Risk 3: Limited compute time**
- If JS2 access is delayed or limited
- **Mitigation**: All experiments can run on Google Colab free tier (T4 GPU)

### Plan B
If original scope proves too ambitious:
1. **Reduce experiments**: Only compare baseline vs. LoRA-8 (skip rank ablation)
2. **Smaller dataset**: Use 10% of training data for faster iteration
3. **Simpler analysis**: Focus on quantitative metrics, minimal error analysis

---

## 8. Milestones

### End of Week 1 (Checkpoint 2: Nov XX)
- ✅ SST-2 data pipeline implemented and tested
- ✅ Baseline model (frozen DistilBERT + classifier) trained
- ✅ Baseline metrics recorded: ~75-82% accuracy
- ✅ Progress note documenting setup and results
- ✅ JS2 environment configured (if access granted)

### End of Week 2 (Main experiments)
- ✅ LoRA implementation using Hugging Face PEFT
- ✅ Experiment 1 complete: Baseline vs. LoRA-8 comparison
- ✅ Experiment 2 in progress: Rank ablation (r=8 vs. r=16)
- ✅ Training curves and comparison tables generated
- ✅ Draft Methods and Results sections for report

### End of Week 3 (Final deliverables)
- ✅ All experiments complete with statistical analysis
- ✅ Error analysis with qualitative examples
- ✅ Final report (3-5 pages, IMRaD format)
- ✅ Clean GitHub repository with README
- ✅ Presentation slides and demo prepared
- ✅ Code fully documented and reproducible

---

## Additional Notes

### Why DistilBERT instead of Llama?
- Open access (no gated model approval needed)
- Smaller size enables faster iteration during development
- Well-established baseline for SST-2 task
- Sufficient for demonstrating LoRA principles

### Alignment with course content
This project directly applies concepts from:
- **Transformers**: Understanding attention mechanisms where LoRA is applied
- **Fine-tuning**: Comparing full fine-tuning vs. PEFT approaches
- **Scaling**: Analyzing parameter efficiency and compute trade-offs
- **Evaluation**: Systematic experimental design with baselines and ablations