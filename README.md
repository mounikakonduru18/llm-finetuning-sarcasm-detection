# Sarcasm Detection in News Headlines: Fine-Tuning BERT and RoBERTa

Fine-tuning `bert-base-uncased` and `roberta-base` to classify English news headlines as sarcastic or non-sarcastic, using the Hugging Face Transformers `Trainer`.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-red) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow) ![Domain](https://img.shields.io/badge/Domain-NLP-green)

## Overview

Sarcasm inverts literal meaning, which breaks conventional sentiment analysis. This project fine-tunes two pre-trained transformer encoders for binary sarcasm classification on a labelled headline dataset. Two notebooks are provided, one per model, built on the same data pipeline.

## Dataset

[News Headlines Dataset for Sarcasm Detection, v2](https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection) (Misra & Arora).

| Property | Value |
|---|---|
| Headlines | 28,619 |
| Sarcastic (The Onion) | 13,634 (47.6%) |
| Non-sarcastic (HuffPost) | 14,985 (52.4%) |
| Train / validation split | 22,895 / 5,724 (80/20, stratified, seed 42) |

Labels are derived from the publication source rather than manual annotation. Please check the dataset's original license before reusing or redistributing the data.

## Methodology

- Tokenisation with the model's own tokenizer, truncated or padded to 128 tokens
- Fine-tuning of all weights with a linear classification head, using the Hugging Face `Trainer` with the AdamW optimiser

| Setting | BERT | RoBERTa |
|---|---|---|
| Base model | `bert-base-uncased` | `roberta-base` |
| Learning rate | 2e-5 | 2e-5 |
| Weight decay | 0.01 | 0.01 |
| Warmup steps | 500 | default |
| Epochs | up to 5, early stopping (patience 2) | 3 |
| Model selection | best validation loss | final epoch |

## Results

| Model | Validation accuracy |
|---|---|
| BERT-base-uncased | 94.16% |

Reported on the 20% validation split. No separate test set was held out.

## Limitations

- Sarcastic and non-sarcastic headlines come from two different publications, so a model can partly learn publication style rather than sarcasm itself.
- The models are trained on headline-style text. Short conversational sarcasm (for example, "I absolutely enjoy getting stuck in traffic every day.") was classified as non-sarcastic in the demo cells.
- Evaluation uses accuracy on a single validation split.

## Usage

The notebooks were developed on Kaggle with a GPU. To run them there, attach the dataset `rmisra/news-headlines-dataset-for-sarcasm-detection` and run all cells in order.

To run locally:

```bash
git clone https://github.com/<mounikakonduru18>/llm-finetuning-sarcasm-detection.git
cd llm-finetuning-sarcasm-detection
pip install transformers datasets accelerate evaluate scikit-learn torch pandas numpy
```

Then change the dataset path in the data-loading cell to the location of `Sarcasm_Headlines_Dataset_v2.json`.

## Project Report

The accompanying report documents the wider study, including comparisons with classical baselines and an embedding-space analysis. The code for those parts is not included in this repository.

## Future Work

- Hold out a separate test set and report precision, recall, F1 and AUC-ROC
- Compare against classical baselines (TF-IDF with Logistic Regression and SVM)
- Evaluate on sarcasm from other domains such as social media
- Try RoBERTa-large and DeBERTa-v3

## Tech Stack

Python, PyTorch, Hugging Face Transformers and Datasets, scikit-learn, pandas, NumPy.

## References

- Misra & Arora, *Sarcasm Detection using News Headlines Dataset*, AI Open, 2023
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers*, NAACL 2019
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, arXiv:1907.11692

## Author

Konduru Mounika, NIT internship project (May 2026 - July 2026)
