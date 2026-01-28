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
```
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
```
---

## How to Run

### A) Python (PyTorch) — BiLSTM & UniLSTM
**File:** `Python Bidirectional and Unidirectional/LSTM_KI.ipynb`

1. **Install dependencies**
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch torchvision torchaudio
   ```

2. **Prepare dataset**
   The notebook contains a splitter that writes:
   - `train.csv`, `val.csv`, `test.csv`

   Set paths in the notebook (top cells), for example:
   ```python
   SINGLE_CSV = "/content/IMDB_Dataset_clean.csv"
   OUT_DIR    = "/content"
   ```

   Run the split cell → it creates:
   - `/content/train.csv`
   - `/content/val.csv`
   - `/content/test.csv`

3. **Train & evaluate**
   The notebook provides separate training sections for:
   - **BiLSTM**
   - **UniLSTM**

   Run one or both.

---

### B) Rust (tch-rs / LibTorch) — BiLSTM & UniLSTM

#### Prerequisites
- Rust toolchain installed (stable)
- LibTorch downloaded automatically by `tch` (internet required)

1. **Prepare `train.csv`, `val.csv`, `test.csv`**
   Generate using the Python notebook splitter, or provide your own CSV files with `text,label`.

2. **Update CSV paths in Rust code**
   Edit the config/path section inside:
   - `Rust Bidirectional/src/main.rs`
   - `Rust Undirectional/src/main.rs`

   Example (Windows path):
   ```rust
   let cfg = Config {
       train: r"C:\path\to\train.csv".into(),
       val:   r"C:\path\to\val.csv".into(),
       test:  r"C:\path\to\test.csv".into(),
       // ...
   };
   ```

3. **Run (release mode recommended)**

   **BiLSTM (Rust):**
   ```bash
   cd "Rust Bidirectional"
   cargo run --release
   ```

   **UniLSTM (Rust):**
   ```bash
   cd "Rust Undirectional"
   cargo run --release
   ```

#### Output
- Each run prints per-epoch train/val metrics + timing, final test metrics, and total runtime.
- Model checkpoints are saved under a `checkpoints/` folder (filename depends on the code config).

---

## Notes (Why BiLSTM Might Not Win Here)
This setup uses **masked average pooling**, which smooths token-position contributions.  
For **document-level sentiment** like IMDB, UniLSTM often already captures enough signal, while BiLSTM adds compute cost due to two-direction processing.

---

## Limitations
- Small hyperparameter search (single configuration) — results may change with tuning.
- CPU-only training makes runtime comparisons hardware-dependent.
- Using masked average pooling may reduce the advantage of “future context” from BiLSTM for this task.
