# 414-legal-document-intelligence

# Legal Document Intelligence: Civil Rights Lawsuit Summarization & Classification

A legal document intelligence system for long civil-rights lawsuits using the Multi-LexSum dataset. This project combines document summarization and legal case classification to analyze lengthy legal documents and provide structured outputs.

## Project Overview

Legal case documents are often extremely long, noisy, and difficult to process manually. This project builds an end-to-end NLP pipeline for:

- Summarizing long legal cases into:
  - **Long summary** (multi-paragraph)
  - **Short summary** (one paragraph)
  - **Tiny summary** (one sentence)

- Predicting:
  - Whether a **class action lawsuit** was sought
  - The **type of legal case**

The project uses the **Multi-LexSum** dataset and focuses on building an interpretable extractive baseline system for long-document intelligence, combining lightweight legal summarization and legal case classification.

---

## Dataset

This project uses:

:contentReference[oaicite:0]{index=0}

The dataset contains:

- **9,280 legal cases**
- Expert-written legal summaries at multiple granularities:
  - long
  - short
  - tiny
- Metadata including:
  - `class_action_sought`
  - `case_type`

Default dataset configuration:

```python
v20230518
```

Example loading:

```python
from datasets import load_dataset

multi_lexsum = load_dataset(
    "allenai/multi_lexsum",
    name="v20230518"
)
```

---

## Repository Structure

```text
414-legal-document-intelligence/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   │
│   └── processed/
│       ├── train_clean.csv
│       ├── validation_clean.csv
│       └── test_clean.csv
│
├── models/
│   ├── class_action/
│   │   └── class_action_tfidf_logreg.pkl
│   │
│   └── case_type/
│       └── case_type_tfidf_logreg.pkl
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_summarization.ipynb
│   └── 03_classification_model.ipynb
│
├── src/
│   ├── load_data.py
│   ├── preprocess.py
│   ├── summarize.py
│   ├── classify.py
│   └── evaluate.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Features

### 1. Legal Case Summarization

A lightweight extractive summarization pipeline for long legal documents.

Pipeline:

```text
Legal document
↓
Text cleaning
↓
Chunking
↓
Chunk filtering
↓
Extractive summarization
↓
Long / Short / Tiny summaries
```

### Summary Types

#### Long Summary
Multi-paragraph extractive summary generated from retrieved chunks.

#### Short Summary
Compressed paragraph-level summary.

#### Tiny Summary
One-sentence summary using lightweight keyword-based sentence ranking.

---

### 2. Legal Case Classification

#### Class Action Prediction

Task:

```text
full_text
→ class_action_sought
```

Labels:

```text
Yes / No
```

Model:

```text
TF-IDF + Logistic Regression
```

Results:

| Metric | Score |
|--------|-------|
| Accuracy | 0.929 |
| Macro F1 | 0.922 |
| Weighted F1 | 0.930 |

---

#### Case Type Classification

Task:

```text
full_text
→ case_type
```

To reduce class imbalance, the model focuses on the top 5 case categories:

- Equal Employment
- Immigration and/or the Border
- Prison Conditions
- Jail Conditions
- Public Benefits / Government Services

Model:

```text
TF-IDF + Logistic Regression
```

Results:

| Metric | Score |
|--------|-------|
| Accuracy | 0.946 |
| Macro F1 | 0.913 |
| Weighted F1 | 0.950 |

---

### 3. Summarization Evaluation

Summaries are evaluated against expert-written references using ROUGE.

Metrics:

- ROUGE-1
- ROUGE-2
- ROUGE-L

Sample-level baseline evaluation using extractive summarization.

Example baseline result:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.342 |
| ROUGE-2 | 0.120 |
| ROUGE-L | 0.196 |

---

## Methods

### Text Cleaning

The preprocessing pipeline removes:

- page headers
- legal formatting artifacts
- OCR noise
- duplicated whitespace
- unnecessary boilerplate

---

### Chunking Strategy

Since legal cases are extremely long, documents are split into overlapping chunks.

Example configuration:

```python
chunk_size = 2500
overlap = 300
```

This allows:

- long document processing
- retrieval-based ranking
- scalable summarization

---

### Extractive Summarization Baseline

Since legal documents are often extremely long, the system uses a lightweight extractive summarization pipeline.

The process:

1. Split long legal documents into overlapping chunks
2. Filter noisy or low-information chunks
3. Extract representative sentences from document chunks
4. Generate long / short / tiny summaries

This lightweight baseline prioritizes interpretability and fast iteration for long legal documents.

---

## Installation

Clone repository:

```bash
git clone <your-repo-url>
cd 414-legal-document-intelligence
```

Create environment:

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Run Summarization

```python
from src.summarize import summarize_document

summary = summarize_document(
    text=case_text,
    summary_type="long"
)

print(summary)
```

Options:

```text
long
short
tiny
```

---

### Run Classification

```python
from src.classify import load_model

model = load_model(
    "models/class_action/class_action_tfidf_logreg.pkl"
)

prediction = model.predict([case_text])

print(prediction)
```

---

## Limitations

Current summaries are:

```text
retrieval-guided extractive baselines
```

Therefore:

- OCR noise may remain
- boilerplate legal language may appear
- summaries are not fully abstractive

