# Financial PhraseBank × DistilBERT Fine-Tuning Study

This repository contains a full experimental notebook for three-way financial sentiment classification on the Financial PhraseBank `sentences_allagree` subset using `distilbert-base-uncased`.

The project starts from the assignment-required fine-tuning regimes, then extends the analysis into multi-round tuning, LoRA-based parameter-efficient fine-tuning, calibration, ensembling, statistical testing, tokenizer ablation, and per-sample failure forensics.

## Notebook

- `financial_phrasebank_distilbert_finetuning_study.ipynb`  
  Main notebook containing the complete experiment, results, discussion, and visual dashboard.

## Project Objective

The goal is not only to maximize a single test score, but to understand what drives performance on a small, finance-domain sentiment dataset:

- How far can a general-domain DistilBERT model go on Financial PhraseBank?
- How do different fine-tuning regimes compare?
- Can LoRA match full-body fine-tuning with far fewer trainable parameters?
- Are later improvements statistically meaningful, or just seed/test-split noise?
- What explains the apparent performance ceiling near 0.95 macro-F1?

## Methods

The notebook compares five model regimes:

1. **Model PT** — pretrained DistilBERT with the default classification setup.
2. **Model PT + FT Classifier** — frozen backbone with fine-tuned classifier head.
3. **Model PT + Own FT Classifier** — custom classifier head over frozen representations.
4. **Model PT + FT Classifier + FT Base** — full-body fine-tuning.
5. **Model PT + FT Classifier + PEFT Base** — LoRA-style parameter-efficient adaptation.

The experiment is organized as a staged investigation:

- **Stage 1:** data loading, stratified splitting, tokenization, metrics, and parameter accounting.
- **Round 1:** minimum viable runs of all five required regimes.
- **Round 2:** evidence-driven adjustments based on Round 1 diagnostics.
- **Round 3:** statistical scaffolding, paired bootstrap checks, and finance-domain benchmark comparison.
- **Round 4:** multi-seed stability, calibration, ensembles, tokenizer/vocabulary ablation, and forensic error analysis.

## Key Results

The strongest single-seed LoRA configuration reached roughly **0.95 test macro-F1**, while the best homogeneous Model 5 ensemble reached **0.9513 test macro-F1**.

Several findings are more important than the leaderboard number:

- **LoRA is highly competitive.** Parameter-efficient adaptation achieves performance close to full-body fine-tuning.
- **The custom head result is diagnostic.** The first custom-head attempt underperformed, but later rounds showed that the issue was training protocol and architecture interaction, not the idea of a custom head itself.
- **Calibration improves confidence quality.** Temperature scaling reduces expected calibration error while leaving accuracy unchanged.
- **Aggregation has a ceiling.** Multi-seed and mega-ensemble variants do not meaningfully exceed the best homogeneous Model 5 ensemble.
- **Statistical uncertainty matters.** A single test split creates a bootstrap confidence interval wide enough that small round-on-round improvements should not be over-claimed.
- **The tokenizer hypothesis is tested and weakened.** Adding finance-specific vocabulary to DistilBERT produces only a small change within seed noise.
- **The remaining failure mode points toward pretraining-domain mismatch.** Per-sample forensic analysis suggests that some errors require finance-discourse knowledge not learned by general-domain pretraining.

## Highlights

This notebook goes beyond a standard fine-tuning assignment in several ways:

- Implements and compares five transformer adaptation regimes in one controlled framework.
- Tracks trainable parameter counts to compare performance and efficiency.
- Uses validation-side diagnostics to motivate later changes instead of tuning blindly.
- Includes multi-seed experiments to estimate stability.
- Applies temperature scaling for calibration analysis.
- Tests ensemble strategies, including homogeneous, heterogeneous, confidence-weighted, stacking, and larger mega-ensembles.
- Runs a controlled vocabulary ablation to test whether token fragmentation explains the performance ceiling.
- Performs per-sample error forensics on universally wrong examples.
- Ends with an honest discussion of what can and cannot be claimed from a small test set.

## Repository Structure

```text
.
├── README.md
└── financial_phrasebank_distilbert_finetuning_study.ipynb
```

## Environment

The notebook installs or imports the main dependencies directly, including:

- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `tqdm`
- `scienceplots`

A GPU is recommended for the full experiment, especially for the multi-round and multi-seed sections.

## How to Run

Open the notebook in Jupyter, JupyterLab, VS Code, or Google Colab, then run cells from top to bottom.

The notebook is designed to be read as an experiment log. Each major section explains:

- what is being tested,
- why the change is being made,
- what result would support or falsify the hypothesis,
- and how to interpret the result.

## Caveats

The reported metrics are based on one 70/15/15 stratified split of a small dataset. The notebook therefore treats small performance deltas cautiously and uses bootstrap analysis and multi-seed experiments to avoid over-claiming.

The project does not claim a definitive causal explanation for the 0.95 macro-F1 ceiling. It does provide evidence that simple vocabulary fragmentation and ensemble aggregation are insufficient explanations, and that finance-domain pretraining is the most plausible next direction to test.
