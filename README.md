# KI — Bidirectional vs Unidirectional LSTM (IMDB Sentiment)
**Exercise 7.4 — Implementing a Bidirectional RNN for Text Classification (Rust + Python)**

This repository implements and compares **Unidirectional LSTM (UniLSTM)** vs **Bidirectional LSTM (BiLSTM)** for **binary text classification (sentiment analysis)** on the **IMDB movie reviews** dataset.  
Implementations are provided in **Rust (tch-rs/LibTorch)** and **Python (PyTorch)** with **matched experiment settings** to ensure a fair comparison.

---

## TL;DR (Main Takeaways)
- **BiLSTM does not significantly improve accuracy** over UniLSTM on IMDB in this setup.
- **BiLSTM is substantially slower**, because it processes sequences in two directions.
- **UniLSTM is the more efficient baseline** here (best accuracy/time trade-off).

---

## Results (From Training Logs / Slides)

| Framework | Direction | Epochs | Test Acc | Precision | Recall | F1 | Total Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| Rust | **Bidirectional** | 10 | 0.846 | 0.845 | 0.846 | 0.843 | 1571.8s |
| Python | **Bidirectional** | 10 | 0.863 | 0.863 | 0.863 | 0.863 | 1982.2s |
| Python | **Unidirectional** | 10 | **0.865** | **0.866** | **0.865** | **0.865** | 1138.1s |
| Rust | **Unidirectional** | 10 | 0.846 | 0.845 | 0.848 | 0.844 | **503.9s** |

---

## Experiment Setup (Matched Across Implementations)

### Data split
- **80% train, 10% validation, 10% test**
- **Seed = 42**

### Preprocessing
- Lowercase
- Tokenization regex: `[A-Za-z0-9]+`
- Vocabulary built **from train only**
- `min_freq = 5`
- Special tokens:
  - `<pad>` = 0
  - `<unk>` = 1
- Sequence length:
  - truncate/pad to `max_len = 128` (**right padding**)

### Models
Both models use **masked average pooling** over LSTM outputs.

#### UniLSTM
- Embedding(128)
- LSTM(hidden=128, layers=1, bidirectional=False)
- Masked AvgPool
- Dropout(0.2)
- FC(hidden → 2 classes)

#### BiLSTM
- Embedding(128)
- LSTM(hidden=128, layers=1, bidirectional=True)
- Masked AvgPool
- Dropout(0.2)
- FC(2×hidden → 2 classes)

### Training
- Loss: Cross Entropy (logits)
- Optimizer: **Adam**, `lr = 1e-3`
- Epochs: **10**
- Batch size: **64**
- Gradient clipping: **1.0**
- Device: **CPU**

### Metrics
- Accuracy
- Precision / Recall / F1
- Total runtime (wall time)

---

## Dataset
**IMDB Movie Reviews** (binary sentiment).

This project expects **CSV files** for training/validation/testing:
- `train.csv`
- `val.csv`
- `test.csv`

### Expected CSV columns
The loaders are flexible, but recommended headers:
- `text` (or `review`)
- `label` (or `sentiment`)

Example:
```csv
text,label
"This movie is great!",positive
"Worst film ever.",negative 
...

---

## Repository Structure

```text
KI-Bidirectional-and-Unidirectional/
├─ Python Bidirectional and Unidirectional/
│  └─ LSTM_KI.ipynb
├─ Rust Bidirectional/
│  ├─ Cargo.toml
│  ├─ Cargo.lock
│  └─ src/main.rs
└─ Rust Undirectional/
   ├─ Cargo.toml
   ├─ Cargo.lock
   └─ src/main.rs
