use anyhow::{Context, Result};
use csv::{ReaderBuilder, StringRecord};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use regex::Regex;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;
use tch::{
    kind::Kind,
    nn,
    nn::OptimizerConfig,
    Device, Tensor,
};
use tch::nn::RNN; // for .seq()

// -------------------- Data structs --------------------
#[derive(Deserialize, Debug, Clone)]
struct Row { text: String, label: String }

#[derive(Clone)]
struct Example { ids: Vec<i64>, len: i64, label: i64 }

struct Vocab {
    stoi: HashMap<String, i64>,
    itos: Vec<String>,
    unk_id: i64,
}
impl Vocab {
    fn new() -> Self {
        let itos = vec!["<pad>".into(), "<unk>".into()];
        let mut stoi = HashMap::new();
        stoi.insert("<pad>".into(), 0);
        stoi.insert("<unk>".into(), 1);
        Self { stoi, itos, unk_id: 1 }
    }
    fn add_token(&mut self, tok: &str) -> i64 {
        if let Some(&id) = self.stoi.get(tok) { id } else {
            let id = self.itos.len() as i64;
            self.stoi.insert(tok.to_string(), id);
            self.itos.push(tok.to_string());
            id
        }
    }
    fn get(&self, tok: &str) -> i64 { *self.stoi.get(tok).unwrap_or(&self.unk_id) }
    fn len(&self) -> i64 { self.itos.len() as i64 }
}
fn simple_tokenize(s: &str, re: &Regex) -> Vec<String> {
    re.find_iter(&s.to_lowercase()).map(|m| m.as_str().to_string()).collect()
}
fn build_vocab(rows: &[Row], min_freq: usize) -> Vocab {
    let re = Regex::new(r"[A-Za-z0-9]+").unwrap();
    let mut freq: HashMap<String, usize> = HashMap::new();
    for r in rows {
        for t in simple_tokenize(&r.text, &re) { *freq.entry(t).or_default() += 1; }
    }
    let mut vocab = Vocab::new();
    let mut pairs: Vec<(String, usize)> = freq.into_iter().collect();
    pairs.sort_by_key(|(_, c)| *c);
    for (tok, c) in pairs { if c >= min_freq { vocab.add_token(&tok); } }
    vocab
}
fn map_labels(rows: &[Row]) -> (BTreeMap<String, i64>, i64) {
    let mut uniq = BTreeMap::<String, i64>::new();
    for r in rows {
        if !uniq.contains_key(&r.label) {
            let id = uniq.len() as i64;
            uniq.insert(r.label.clone(), id);
        }
    }
    (uniq.clone(), uniq.len() as i64)
}
fn rows_to_examples(rows: &[Row], vocab: &Vocab, labmap: &BTreeMap<String, i64>, max_len: usize) -> Vec<Example> {
    let re = Regex::new(r"[A-Za-z0-9]+").unwrap();
    rows.iter().map(|r| {
        let mut ids: Vec<i64> = simple_tokenize(&r.text, &re).into_iter().map(|t| vocab.get(&t)).collect();
        if ids.len() > max_len { ids.truncate(max_len); }
        let len = ids.len() as i64;
        Example { ids, len, label: *labmap.get(&r.label).unwrap() }
    }).collect()
}

// -------------------- CSV helpers --------------------
fn detect_delimiter<P: AsRef<Path>>(path: P) -> u8 {
    let f = match fs::File::open(path) { Ok(f) => f, Err(_) => return b',' };
    let mut reader = BufReader::new(f);
    let mut first_line = String::new();
    if reader.read_line(&mut first_line).is_ok() {
        let commas = first_line.matches(',').count();
        let semis  = first_line.matches(';').count();
        if semis > commas { b';' } else { b',' }
    } else { b',' }
}
fn read_csv_flex(path: &str) -> Result<Vec<Row>> {
    let delim = detect_delimiter(path);
    let mut rdr = ReaderBuilder::new()
        .has_headers(true).flexible(true).delimiter(delim)
        .from_path(path).with_context(|| format!("reading {}", path))?;
    let headers = rdr.headers()?.clone();
    let (mut ti, mut li) = resolve_text_label_idx(&headers);
    if headers.len() < 2 { ti = 0; li = usize::MAX; }
    let mut out = vec![];
    for rec in rdr.records() {
        let r: StringRecord = rec?;
        let len = r.len(); if len == 0 { continue; }
        let (text, label) = if len >= 2 {
            if li != usize::MAX && li < len && ti < len {
                (r.get(ti).unwrap_or("").to_string(), r.get(li).unwrap_or("").to_string())
            } else {
                let label = r.get(len - 1).unwrap_or("").to_string();
                let mut parts: Vec<&str> = Vec::with_capacity(len - 1);
                for i in 0..(len - 1) { parts.push(r.get(i).unwrap_or("")); }
                (parts.join(","), label)
            }
        } else { (r.get(0).unwrap_or("").to_string(), String::new()) };
        if !text.is_empty() && !label.is_empty() { out.push(Row { text, label }); }
    }
    Ok(out)
}
fn resolve_text_label_idx(h: &StringRecord) -> (usize, usize) {
    let mut ti = 0usize; let mut li = 1usize;
    for (i, name) in h.iter().enumerate() {
        let n = name.trim().trim_start_matches('\u{feff}').to_lowercase();
        if n == "text" || n == "review" || n == "sentence" { ti = i; }
        if n == "label" || n == "sentiment" || n == "target" { li = i; }
    }
    (ti, li)
}

// -------------------- Batching --------------------
struct Batch { x: Tensor, lens: Tensor, y: Tensor }
fn make_batches(
    ex: &[Example], pad_id: i64, batch_size: usize, device: Device, max_len: usize, seed: u64
) -> Vec<Batch> {
    let mut idx: Vec<usize> = (0..ex.len()).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    idx.shuffle(&mut rng);
    let mut batches = vec![];
    for chunk in idx.chunks(batch_size) {
        let b = chunk.len();
        let mut x = Vec::<i64>::with_capacity(b * max_len);
        let mut lens = Vec::<i64>::with_capacity(b);
        let mut y = Vec::<i64>::with_capacity(b);
        for &i in chunk {
            let mut ids = ex[i].ids.clone();
            lens.push(ex[i].len);
            if ids.len() < max_len { ids.resize(max_len, pad_id); }
            x.extend(ids);
            y.push(ex[i].label);
        }
        let x = Tensor::from_slice(&x).view([b as i64, max_len as i64]).to_device(device);
        let lens = Tensor::from_slice(&lens).to_device(device);
        let y = Tensor::from_slice(&y).to_device(device);
        batches.push(Batch { x, lens, y });
    }
    batches
}

// -------------------- Model: UNIdirectional LSTM --------------------
struct UniLstmClf {
    embed: nn::Embedding,
    lstm: nn::LSTM,
    fc: nn::Linear,
    dropout_p: f64,
}
impl UniLstmClf {
    fn new(
        vs: &nn::Path, vocab_size: i64, emb_dim: i64, hidden: i64, num_layers: i64,
        num_classes: i64, dropout_p: f64,
    ) -> Self {
        let embed = nn::embedding(vs / "emb", vocab_size, emb_dim, Default::default());
        let lstm_cfg = nn::RNNConfig { bidirectional: false, num_layers, ..Default::default() };
        let lstm = nn::lstm(vs / "lstm", emb_dim, hidden, lstm_cfg);
        let fc = nn::linear(vs / "fc", hidden, num_classes, Default::default());
        Self { embed, lstm, fc, dropout_p }
    }
    fn forward(&self, xs: &Tensor, lens: &Tensor, train: bool) -> Tensor {
        let seq_len = xs.size()[1];
        let mut emb = xs.apply(&self.embed);
        if train && self.dropout_p > 0.0 { emb = emb.dropout(self.dropout_p, train); }
        // LSTM expects (seq, batch, feat)
        let emb_t = emb.transpose(0, 1);
        let (out, _hc) = self.lstm.seq(&emb_t);
        let out = out.transpose(0, 1); // (batch, seq, hidden)
        // average pooling with mask
        let rng = Tensor::arange(seq_len, (Kind::Int64, xs.device())).unsqueeze(0);
        let mask = rng.lt_tensor(&lens.unsqueeze(1)).to_kind(Kind::Float);
        let masked = &out * &mask.unsqueeze(-1);
        let sum_t = masked.sum_dim_intlist([1].as_slice(), false, Kind::Float);
        let mut pooled = &sum_t / &lens.to_kind(Kind::Float).clamp_min(1.0).unsqueeze(1);
        if train && self.dropout_p > 0.0 { pooled = pooled.dropout(self.dropout_p, train); }
        pooled.apply(&self.fc)
    }
}

// -------------------- Metrics --------------------
fn accuracy(logits: &Tensor, y: &Tensor) -> f64 {
    let pred = logits.argmax(-1, false);
    pred.eq_tensor(y).to_kind(Kind::Float).mean(Kind::Float).double_value(&[])
}
fn f1_macro(logits: &Tensor, y: &Tensor, num_classes: i64) -> f64 {
    let pred = logits.argmax(-1, false);
    let mut f1s = vec![];
    for c in 0..num_classes {
        let c_t = Tensor::from(c).to_device(y.device());
        let tp = pred.eq_tensor(&c_t).logical_and(&y.eq_tensor(&c_t)).to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
        let fp = pred.eq_tensor(&c_t).logical_and(&y.ne_tensor(&c_t)).to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
        let fn_ = pred.ne_tensor(&c_t).logical_and(&y.eq_tensor(&c_t)).to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
        let prec = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
        let rec  = if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) };
        let f1 = if prec + rec == 0.0 { 0.0 } else { 2.0 * prec * rec / (prec + rec) };
        f1s.push(f1);
    }
    f1s.iter().sum::<f64>() / (num_classes as f64)
}
fn precision_recall_macro(logits: &Tensor, y: &Tensor, num_classes: i64) -> (f64, f64) {
    let pred = logits.argmax(-1, false);
    let mut precisions = vec![];
    let mut recalls = vec![];
    for c in 0..num_classes {
        let c_t = Tensor::from(c).to_device(y.device());
        let tp = pred.eq_tensor(&c_t).logical_and(&y.eq_tensor(&c_t)).to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
        let fp = pred.eq_tensor(&c_t).logical_and(&y.ne_tensor(&c_t)).to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
        let fn_ = pred.ne_tensor(&c_t).logical_and(&y.eq_tensor(&c_t)).to_kind(Kind::Float).sum(Kind::Float).double_value(&[]);
        let p = if tp + fp == 0.0 { 0.0 } else { tp / (tp + fp) };
        let r = if tp + fn_ == 0.0 { 0.0 } else { tp / (tp + fn_) };
        precisions.push(p);
        recalls.push(r);
    }
    let p_macro = precisions.iter().sum::<f64>() / (num_classes as f64);
    let r_macro = recalls.iter().sum::<f64>() / (num_classes as f64);
    (p_macro, r_macro)
}

// -------------------- Config & main --------------------
struct Config {
    train: String, val: String, test: String,
    max_len: usize, batch_size: usize, epochs: usize, lr: f64,
    emb_dim: i64, hidden: i64, num_layers: i64, dropout: f64,
    min_freq: usize, seed: u64,
}

fn main() -> Result<()> {
    // === EDIT PATHS DI SINI ===
    let cfg = Config {
        train: r"C:\Rust7\Project\train.csv".into(),
        val:   r"C:\Rust7\Project\val.csv".into(),
        test:  r"C:\Rust7\Project\test.csv".into(),
        // ========================
        max_len: 128, batch_size: 64, epochs: 10, lr: 1e-3,
        emb_dim: 128, hidden: 128, num_layers: 1, dropout: 0.2,
        min_freq: 5, seed: 42,
    };

    tch::manual_seed(cfg.seed as i64);
    let device = Device::cuda_if_available();
    eprintln!("Device: {:?} | Mode: UNIdirectional | layers={} hidden={}", device, cfg.num_layers, cfg.hidden);

    let t_total = Instant::now();

    // baca 3 file
    let train_rows = read_csv_flex(&cfg.train).context("read train.csv")?;
    let val_rows   = read_csv_flex(&cfg.val).context("read val.csv")?;
    let test_rows  = read_csv_flex(&cfg.test).context("read test.csv")?;
    anyhow::ensure!(!train_rows.is_empty() && !val_rows.is_empty() && !test_rows.is_empty(), "train/val/test kosong?");
    eprintln!("→ train: {}, val: {}, test: {}", train_rows.len(), val_rows.len(), test_rows.len());

    // vocab & label map
    let vocab = build_vocab(&train_rows, cfg.min_freq);
    let (labmap, num_classes) = map_labels(&train_rows);
    eprintln!("Vocab size: {}", vocab.len());
    eprintln!("Labels: {:?}", labmap);

    // examples
    let train_ex = rows_to_examples(&train_rows, &vocab, &labmap, cfg.max_len);
    let val_ex   = rows_to_examples(&val_rows,  &vocab, &labmap, cfg.max_len);
    let test_ex  = rows_to_examples(&test_rows, &vocab, &labmap, cfg.max_len);

    // model
    let vs = nn::VarStore::new(device);
    let root = &vs.root();
    let model = UniLstmClf::new(root, vocab.len(), cfg.emb_dim, cfg.hidden, cfg.num_layers, num_classes, cfg.dropout);
    let mut opt = nn::Adam::default().build(&vs, cfg.lr)?;

    // -------- train loop --------
    for epoch in 1..=cfg.epochs {
        let t_epoch = Instant::now();

        // TRAIN
        let t_train = Instant::now();
        let epoch_seed = cfg.seed.wrapping_add(epoch as u64);
        let mut train_batches = make_batches(&train_ex, 0, cfg.batch_size, device, cfg.max_len, epoch_seed);
        let mut tr_loss = 0.0f64; let mut tr_acc = 0.0f64; let mut n_b = 0usize;
        for b in train_batches.drain(..) {
            let logits = model.forward(&b.x, &b.lens, true);
            let loss = logits.cross_entropy_for_logits(&b.y);
            opt.backward_step(&loss);
            tr_loss += loss.double_value(&[]);
            tr_acc  += accuracy(&logits, &b.y);
            n_b += 1;
        }
        let train_secs = t_train.elapsed().as_secs_f32();

        // VAL
        let t_val = Instant::now();
        let val_batches = make_batches(&val_ex, 0, cfg.batch_size, device, cfg.max_len, cfg.seed);
        let mut v_loss = 0f64; let mut v_acc = 0f64; let mut v_f1 = 0f64;
        let mut v_prec = 0f64; let mut v_rec = 0f64; let mut v_b = 0usize;
        for b in val_batches {
            let logits = model.forward(&b.x, &b.lens, false);
            let loss = logits.cross_entropy_for_logits(&b.y);
            v_loss += loss.double_value(&[]);
            v_acc  += accuracy(&logits, &b.y);
            v_f1   += f1_macro(&logits, &b.y, num_classes);
            let (pm, rm) = precision_recall_macro(&logits, &b.y, num_classes);
            v_prec += pm; v_rec += rm;
            v_b += 1;
        }
        let val_secs = t_val.elapsed().as_secs_f32();

        let epoch_secs = t_epoch.elapsed().as_secs_f32();
        println!(
            "Epoch {:02} | train loss {:.4} acc {:.3} | val loss {:.4} acc {:.3} P {:.3} R {:.3} F1 {:.3} | time train {:.1}s val {:.1}s total {:.1}s",
            epoch, tr_loss / n_b as f64, tr_acc / n_b as f64,
            v_loss / v_b as f64, v_acc / v_b as f64,
            v_prec / v_b as f64, v_rec / v_b as f64, v_f1 / v_b as f64,
            train_secs, val_secs, epoch_secs
        );
    }

    // -------- test --------
    let t_test = Instant::now();
    let test_batches = make_batches(&test_ex, 0, cfg.batch_size, device, cfg.max_len, cfg.seed);
    let mut t_acc = 0f64; let mut t_f1 = 0f64; let mut t_prec = 0f64; let mut t_rec = 0f64; let mut t_b = 0usize;
    for b in test_batches {
        let logits = model.forward(&b.x, &b.lens, false);
        t_acc  += accuracy(&logits, &b.y);
        t_f1   += f1_macro(&logits, &b.y, num_classes);
        let (pm, rm) = precision_recall_macro(&logits, &b.y, num_classes);
        t_prec += pm; t_rec += rm;
        t_b += 1;
    }
    let test_secs = t_test.elapsed().as_secs_f32();
    println!(
        "TEST | acc {:.3} | P {:.3} R {:.3} F1 {:.3} | time {:.1}s",
        t_acc / t_b as f64, t_prec / t_b as f64, t_rec / t_b as f64, t_f1 / t_b as f64, test_secs
    );

    println!("TOTAL time {:.1}s", t_total.elapsed().as_secs_f32());

    fs::create_dir_all("checkpoints").ok();
    nn::VarStore::save(&vs, "checkpoints/lstm_uni_textclf.ot")?;
    Ok(())
}
