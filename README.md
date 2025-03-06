# Automated Content Moderation Using Machine Learning

## Overview

In the digital age, managing online discourse is a growing challenge due to the prevalence of toxic and harmful content. This project focuses on developing a machine learning pipeline to detect and classify toxic comments with high accuracy. We implemented multiple models, from traditional classifiers to deep learning architectures, to identify toxicity across different categories.

---

## Dataset

The dataset consists of Wikipedia comments labeled by human raters for toxicity types:

- **Toxic**
- **Severe Toxic**
- **Obscene**
- **Threat**
- **Insult**
- **Identity Hate**

---

## Data Preprocessing

- **Stopword Removal:** Eliminated unnecessary words using NLTK.
- **Text Cleaning:** Removed URLs, special characters, and extra whitespaces via regex.
- **Feature Extraction:** Utilized TF-IDF vectorization and GPT-based embedding techniques.
- **Class Balancing:** Addressed imbalance using `LeastSampledClassSampler` to ensure fair model training.

---

## Models Implemented

### 1. Traditional Machine Learning Classifiers

#### Naïve Bayes Classifier
- Implemented **Multinomial Naïve Bayes** with `OneVsRestClassifier`.
- **Performance:** Struggled with minority classes, resulting in low F1 scores.

#### Logistic Regression
- Used **TF-IDF features** with `OneVsRestClassifier`.
- **Performance:** Improved over Naïve Bayes, yet failed to capture complexity in severe toxicity categories.

#### Linear Support Vector Classifier (SVC)
- **Performance:** Performed significantly better than previous models.
- Best **F1 scores** in detecting general toxicity.

### 2. Deep Learning Models

#### Custom Neural Network
- **Architecture:** Three linear layers with **ReLU activations**.
- **Loss Function:** `BCEWithLogitsLoss` for multi-label classification.
- **Training:** Used **AdamW optimizer** with a learning rate of `0.001`.
- **Performance:** Much better than traditional models but struggled with minority classes.

#### Pre-trained Transformers (BERT-based Model)
- **Architecture:** Fine-tuned **BERT model** using `BertForMultiLabelSequenceClassification`.
- **Tokenizer:** `BertTokenizer` from Hugging Face.
- **Training:** 5 epochs, **AdamW optimizer**, learning rate `1e-5`.
- **Performance:**
  - **Best results overall**, outperforming all previous models.
  - Struggled with **identity hate** and **severe toxicity**, similar to the neural network.

---

## Performance Comparison

### Macro-Averaged F1 Scores (Test Data)

| Model              | Toxic | Severe Toxic | Obscene | Threat | Insult | Identity Hate |
|--------------------|-------|-------------|---------|--------|--------|---------------|
| **Naïve Bayes**   | 0.28  | 0.00        | 0.19    | 0.00   | 0.07   | 0.00          |
| **Logistic Regression** | 0.73  | 0.35        | 0.73    | 0.24   | 0.63   | 0.24          |
| **Linear SVC**    | 0.77  | 0.37        | 0.78    | 0.36   | 0.68   | 0.37          |
| **Neural Network** | 0.85  | 0.66        | 0.81    | 0.55   | 0.80   | 0.73          |
| **BERT Transformer** | 0.96  | 0.41        | 0.83    | 0.58   | 0.78   | 0.73          |

---

## Results and Insights

- **Traditional classifiers** were not effective for minority classes.
- **Neural networks** and **BERT** significantly improved performance, particularly for common toxic categories.
- **BERT performed the best**, but still struggles with highly imbalanced data.

---

## Future Work

- Fine-tune **hyperparameters** for transformers.
- Explore **ensemble models** combining deep learning with traditional classifiers.
- Improve **dataset balance** using advanced oversampling techniques.
- Test alternative architectures like **RoBERTa** or **DistilBERT** for better efficiency.

---

## Installation & Usage

### Requirements

- Python 3.x
- Jupyter Notebook / Google Colab
- Libraries: `pandas`, `numpy`, `nltk`, `scikit-learn`, `torch`, `transformers`

### Setup

```bash
pip install -r requirements.txt
```

### Run the Project

```bash
jupyter notebook
```

Open Each Notebook and Run it. To obtain the original dataset for this project, you can email at eessashahid@gmail.com

---

