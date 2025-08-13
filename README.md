# Automated Content Moderation with Machine Learning and Transformers

Detecting and mitigating toxic content at scale requires models that are accurate, robust, and reproducible. This repository implements an end‑to‑end workflow for multi‑label toxic comment classification, from classical baselines to a fine‑tuned BERT model, with clear guidance to reproduce results and extend the work.

## Highlights

- Multi‑label classification for six toxicity categories with strong baselines and a fine‑tuned transformer
- Robust preprocessing, TF‑IDF features, and optional embedding-based features
- Class imbalance handling via sampling strategies and careful metric selection
- Reproducible notebooks for each stage, with documented configurations and results
- Clear path to extend to new models (e.g., RoBERTa, DistilBERT) and datasets

## Repository Structure

```
MLContentModeration/
  G38_NB1.ipynb    # Baselines: TF‑IDF + Naive Bayes, Logistic Regression, Linear SVC
  G38_NB2.ipynb    # Intermediate exploration and analysis; setup for advanced models
  G38_NB3.ipynb    # Transformer fine‑tuning (BERT) for multi‑label classification
  G38_Report.pdf   # Project report with methodology and results
  bert.png         # Illustration used in NB3
  naive.png        # Illustration used in NB1
  intermediates/   # Build artifacts (not required to run notebooks)
```

## Dataset

This project targets the Jigsaw Toxic Comment Classification dataset (Wikipedia talk page comments) with the following labels:

- Toxic
- Severe toxic
- Obscene
- Threat
- Insult
- Identity hate

Download the dataset from Kaggle and place the files as follows:

```
MLContentModeration/
  train.csv
  data/
    test.csv
    test_labels.csv
```

Adjust paths in the notebooks if you use a different layout.

## Environment and Setup

- Python 3.9+ recommended
- GPU highly recommended for transformer fine‑tuning (CUDA/CuDNN)

Install core dependencies (create a virtualenv or conda env first):

```bash
pip install pandas numpy scikit-learn nltk matplotlib torch torchvision torchaudio transformers tqdm jupyter
```

Optional components used in advanced cells of NB3:

```bash
pip install tiktoken gpt4all pytorch-multilabel-balanced-sampler
```

If you use NLTK stopwords or tokenizers, ensure resources are available:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

Launch notebooks:

```bash
jupyter notebook
```

Open `G38_NB1.ipynb` → `G38_NB2.ipynb` → `G38_NB3.ipynb` sequentially.

## Methodology

### Preprocessing

- Lowercasing, URL removal, non‑alphanumeric stripping, whitespace normalization
- Stopword removal using NLTK
- Feature extraction: TF‑IDF (baselines) and tokenization for BERT (`BertTokenizer`)
- Optional: embedding utilities (`GPT2Tokenizer`, `tiktoken`, `gpt4all.Embed4All`) for experimentation

### Handling Class Imbalance

- Exploratory analysis of label distribution (NB1)
- Strategies explored in NB3:
  - Downsampling majority class to match positive samples for speed/balance
  - Optional samplers (e.g., `LeastSampledClassSampler`) for balanced batching

### Models

1) Classical baselines (NB1)
- Multinomial Naive Bayes + One‑Vs‑Rest
- Logistic Regression + One‑Vs‑Rest
- Linear SVC + One‑Vs‑Rest

2) Neural baselines (notebook narrative)
- Shallow feed‑forward network with ReLU and `BCEWithLogitsLoss`

3) Transformer fine‑tuning (NB3)
- `BertForSequenceClassification` adapted for multi‑label classification
- Tokenization with `BertTokenizer` (uncased), typical max length ≈ 200 tokens based on sequence analysis
- Optimization with AdamW; learning rate around 1e‑5; 5 epochs (configurable)
- Loss: `BCEWithLogitsLoss` with sigmoid outputs per class

### Evaluation

- Multi‑label metrics via scikit‑learn: precision, recall, F1 (macro/micro), classification reports
- Primary focus on macro‑F1 due to class imbalance
- Per‑class confusion/misclassification patterns analyzed in NB1/NB3

## Results (Representative)

Macro‑averaged F1 on held‑out data across labels:

| Model | Toxic | Severe Toxic | Obscene | Threat | Insult | Identity Hate |
|---|---:|---:|---:|---:|---:|---:|
| Naive Bayes | 0.28 | 0.00 | 0.19 | 0.00 | 0.07 | 0.00 |
| Logistic Regression | 0.73 | 0.35 | 0.73 | 0.24 | 0.63 | 0.24 |
| Linear SVC | 0.77 | 0.37 | 0.78 | 0.36 | 0.68 | 0.37 |
| Neural Network | 0.85 | 0.66 | 0.81 | 0.55 | 0.80 | 0.73 |
| BERT (fine‑tuned) | 0.96 | 0.41 | 0.83 | 0.58 | 0.78 | 0.73 |

Notes
- BERT consistently delivers the strongest overall performance but still struggles on the rarest classes.
- Linear SVC is a strong traditional baseline with competitive performance and fast training.

For detailed plots, confusion matrices, and extended commentary, see the notebooks and `G38_Report.pdf`.

## How to Reproduce

1) Download dataset and place CSVs as described above.
2) Install dependencies and NLTK resources.
3) Run notebooks in order:
   - `G38_NB1.ipynb`: preprocessing, EDA, TF‑IDF + baseline models
   - `G38_NB2.ipynb`: intermediate exploration and setup for deep models
   - `G38_NB3.ipynb`: BERT fine‑tuning and evaluation
4) Adjust hyperparameters in the training cells (epochs, learning rate, batch size, max sequence length) and re‑run.

Reproducibility tips
- `random_state=42` used in splits; fix seeds for `numpy`/`torch` if you need exact runs.
- GPU/driver versions can influence training speed and minor metric variance.

## Extending This Project

- Swap encoder: RoBERTa (`roberta-base`), DistilBERT, DeBERTa
- Tune class weights or use focal loss for extreme imbalance
- Robust text normalization (emoji, misspellings), contextual data augmentation
- Ensembling: blend transformer logits with classical models
- Export to TorchScript/ONNX for deployment

## Ethical Use and Limitations

- Toxicity is context‑dependent; models can reflect annotation and data biases.
- Minority and rare classes remain challenging; decisions should not be fully automated without human oversight.
- Always measure fairness across sub‑groups when deploying in production settings.

## Citations

- Jigsaw Toxic Comment Classification dataset (Kaggle)
- Devlin et al., BERT: Pre‑training of Deep Bidirectional Transformers for Language Understanding
- Wolf et al., Hugging Face Transformers

## License and Acknowledgments

This repository is for educational and research purposes. See `G38_Report.pdf` for a detailed write‑up of methodology and findings. Acknowledge and follow the terms of the dataset provider when using the data.
