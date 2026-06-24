# EegToText-project

import os, csv, random
import numpy as np
import scipy.io
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (BartForConditionalGeneration,
                          BartTokenizer)
from transformers.modeling_outputs import BaseModelOutput

CSV_FOLDER = '/home/ubuntu/eeg/'
EEG_FOLDER = '/home/ubuntu/raw_eeg/'
SAVE_PATH  = 'eeg_bart_model.pt'
LOG_PATH   = 'training_log.csv'

N_CHANNELS = 128
SEQ_LEN    = 512
D_MODEL    = 256
N_PREFIX   = 4
BART_DIM   = 768
MAX_SEQ    = 48
BATCH_SIZE = 8
EPOCHS     = 300
LR         = 3e-4
PATIENCE   = 40
MIN_DELTA  = 0.005

DEVICE = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# LOAD DATA CORRECTLY
# Key fix: pair each sentence with MULTIPLE EEG windows
# not just one window per sentence

all_eeg, all_sentences = [], []

for i in range(1, 8):
    csv_path = os.path.join(CSV_FOLDER, f'nr_{i}.csv')
    eeg_path = os.path.join(EEG_FOLDER,
                            f'YAC_NR{i}_EEG.mat')

    if not os.path.exists(csv_path) or \
       not os.path.exists(eeg_path):
        print(f"Skipping NR{i}")
        continue

    # load sentences
    df        = pd.read_csv(csv_path, sep=';',
                            header=None)
    sentences = df.iloc[:, 2].dropna().tolist()
    # keep duplicates — each row = one sentence reading
    print(f"NR{i} -> {len(sentences)} sentence rows")

    # load EEG
    mat        = scipy.io.loadmat(eeg_path)
    eeg_struct = mat['EEG']
    eeg_raw    = np.array(
        eeg_struct['data'][0, 0]).astype(np.float32)

    if eeg_raw.shape[0] != N_CHANNELS:
        eeg_raw = eeg_raw.T

    # normalize per channel
    mean    = eeg_raw.mean(axis=1, keepdims=True)
    std     = eeg_raw.std(axis=1,  keepdims=True) + 1e-8
    eeg_raw = (eeg_raw - mean) / std

    total_time = eeg_raw.shape[1]
    n_windows  = total_time // SEQ_LEN
    print(f"NR{i} -> {n_windows} EEG windows "
          f"from {total_time} timepoints")

    # pair each sentence with corresponding EEG window
    n_pairs = min(n_windows, len(sentences))
    for j in range(n_pairs):
        chunk = eeg_raw[:, j*SEQ_LEN:(j+1)*SEQ_LEN]
        sent  = str(sentences[j]).strip()
        if len(sent) > 5:  # skip empty/short sentences
            all_eeg.append(chunk.astype(np.float32))
            all_sentences.append(sent)

    print(f"NR{i} -> {n_pairs} pairs created\n")

print(f"Total pairs     : {len(all_eeg)}")
print(f"Unique sentences: {len(set(all_sentences))}\n")
assert len(all_eeg) > 0, "No data loaded!"


# SPLIT BY SENTENCE — NO OVERLAP 
unique_sents = list(set(all_sentences))
random.shuffle(unique_sents)
split = int(0.8 * len(unique_sents))
train_sent_set = set(unique_sents[:split])
val_sent_set   = set(unique_sents[split:])

train_eeg, train_sents = [], []
val_eeg,   val_sents   = [], []

for eeg, sent in zip(all_eeg, all_sentences):
    if sent in train_sent_set:
        train_eeg.append(eeg)
        train_sents.append(sent)
    else:
        val_eeg.append(eeg)
        val_sents.append(sent)

print(f"Train: {len(train_eeg)} pairs "
      f"({len(train_sent_set)} unique sents)")
print(f"Val  : {len(val_eeg)} pairs "
      f"({len(val_sent_set)} unique sents)\n")


# LIGHT AUGMENTATION ON TRAIN ONLY 
def augment(eeg_list, sent_list, times=3):
    aug_eeg, aug_sents = [], []
    for eeg, sent in zip(eeg_list, sent_list):
        aug_eeg.append(eeg)
        aug_sents.append(sent)
        for _ in range(times - 1):
            e = eeg.copy()
            e += np.random.randn(
                     *e.shape).astype(np.float32) * 0.01
            aug_eeg.append(e)
            aug_sents.append(sent)
    return aug_eeg, aug_sents

train_eeg, train_sents = augment(
    train_eeg, train_sents, times=3)
print(f"After aug: {len(train_eeg)} train pairs\n")


# dataset
class EEGDataset(Dataset):
    def __init__(self, eeg_list, sent_list):
        self.eeg   = eeg_list
        self.sents = sent_list
    def __len__(self):
        return len(self.eeg)
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.eeg[idx]),
                self.sents[idx])

train_ds = EEGDataset(train_eeg,  train_sents)
val_ds   = EEGDataset(val_eeg,    val_sents)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      shuffle=True,  drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                      shuffle=False, drop_last=False)
print(f"Train batches: {len(train_dl)} | "
      f"Val batches: {len(val_dl)}")


# EEG encoder
class EEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # spatial filter across channels
        self.spatial = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        # temporal CNN — 512 -> 128 timesteps
        self.temporal = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Conv1d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        # transformer — better than LSTM for this size
        enc_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.25,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=3)

        self.proj = nn.Linear(128, D_MODEL)
        self.norm = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        # x: (B, 128, 512)
        x = self.spatial(x)      # (B, 64, 512)
        x = self.temporal(x)     # (B, 128, 64)
        x = x.permute(0, 2, 1)  # (B, 64, 128)
        x = self.transformer(x) # (B, 64, 128)
        x = x.mean(dim=1)       # (B, 128)
        return self.norm(self.proj(x))  # (B, D_MODEL)


#  ADAPTER 
class Adapter(nn.Module)
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, BART_DIM * N_PREFIX),
            nn.LayerNorm(BART_DIM * N_PREFIX)
        )
    def forward(self, x):
        return self.net(x).view(-1, N_PREFIX, BART_DIM)


# LOAD BART 
print("\nLoading BART...")
tokenizer = BartTokenizer.from_pretrained(
                'facebook/bart-base')
bart = BartForConditionalGeneration.from_pretrained(
           'facebook/bart-base').to(DEVICE)

# freeze all BART
for param in bart.parameters():
    param.requires_grad = False

# unfreeze last 2 decoder layers + lm_head
for layer in bart.model.decoder.layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True
for param in bart.lm_head.parameters():
    param.requires_grad = True

eeg_encoder = EEGEncoder().to(DEVICE)
adapter     = Adapter().to(DEVICE)

n = sum(p.numel() for p in
        list(eeg_encoder.parameters()) +
        list(adapter.parameters()) +
        [p for p in bart.parameters()
         if p.requires_grad])
print(f"Trainable: {n/1e6:.2f}M params\n")

optimizer = optim.AdamW([
    {'params': eeg_encoder.parameters(), 'lr': 3e-4},
    {'params': adapter.parameters(),     'lr': 3e-4},
    {'params': [p for p in bart.parameters()
                if p.requires_grad],     'lr': 1e-5},
], weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6)


# LOSS
def compute_loss(eeg_batch, sent_batch):
    B = eeg_batch.size(0)

    feat   = eeg_encoder(eeg_batch)
    prefix = adapter(feat)
    pmask  = torch.ones(B, N_PREFIX,
                        dtype=torch.long,
                        device=DEVICE)

    # FIXED tokenizer — use bos/eos correctly for BART
    tokens = tokenizer(
        list(sent_batch),
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=MAX_SEQ,
        add_special_tokens=True
    ).to(DEVICE)

    enc_out = BaseModelOutput(
        last_hidden_state=prefix)

    out = bart(
        encoder_outputs=enc_out,
        attention_mask=pmask,
        labels=tokens['input_ids'],
        decoder_attention_mask=tokens['attention_mask']
    )
    return out.loss


# GENERATE TEXT 
def generate(eeg_array):
    eeg_encoder.eval(); adapter.eval(); bart.eval()
    with torch.no_grad():
        t      = torch.FloatTensor(
                     eeg_array).unsqueeze(0).to(DEVICE)
        feat   = eeg_encoder(t)
        prefix = adapter(feat)
        pmask  = torch.ones(1, N_PREFIX,
                            dtype=torch.long).to(DEVICE)
        enc_out = BaseModelOutput(
            last_hidden_state=prefix)
        out = bart.generate(
            encoder_outputs=enc_out,
            attention_mask=pmask,
            max_new_tokens=60,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            length_penalty=1.0,
            early_stopping=True,
            forced_bos_token_id=tokenizer.bos_token_id
        )
    return tokenizer.decode(
               out[0],
               skip_special_tokens=True).strip()


#  TRAINING LOOP 
print(f"Training {EPOCHS} epochs...\n")
log_rows         = []
best_val_loss    = float('inf')
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    # TRAIN
    eeg_encoder.train(); adapter.train(); bart.train()
    train_loss = 0.0
    for eeg_b, sent_b in train_dl:
        eeg_b = eeg_b.to(DEVICE)
        optimizer.zero_grad()
        loss = compute_loss(eeg_b, sent_b)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(eeg_encoder.parameters()) +
            list(adapter.parameters()) +
            [p for p in bart.parameters()
             if p.requires_grad], 1.0)
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)

    # VALIDATE
    eeg_encoder.eval(); adapter.eval(); bart.eval()
    val_loss = 0.0
    with torch.no_grad():
        for eeg_b, sent_b in val_dl:
            eeg_b = eeg_b.to(DEVICE)
            val_loss += compute_loss(
                            eeg_b, sent_b).item()
    val_loss /= len(val_dl)
    scheduler.step()

    gap = val_loss - train_loss
    flag = ("GREAT" if gap < 1.0 else
            "OK"    if gap < 2.0 else "OVERFIT")
    lr  = optimizer.param_groups[0]['lr']

    log_rows.append([epoch,
                     round(train_loss, 4),
                     round(val_loss, 4),
                     round(gap, 4)])
    print(f"Epoch {epoch:3d}/{EPOCHS} | "
          f"Train: {train_loss:.4f} | "
          f"Val: {val_loss:.4f} | "
          f"Gap: {gap:.2f} {flag} | "
          f"LR: {lr:.6f}")

    # save best
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss    = val_loss
        patience_counter = 0
        torch.save({
            'eeg_encoder': eeg_encoder.state_dict(),
            'adapter'    : adapter.state_dict(),
            'epoch'      : epoch,
            'val_loss'   : val_loss,
            'n_channels' : N_CHANNELS,
            'n_prefix'   : N_PREFIX,
            'd_model'    : D_MODEL,
        }, SAVE_PATH)
        print(f"  [SAVED] val={val_loss:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stop at epoch {epoch}")
            break

    # sample every 25 epochs
    if epoch % 25 == 0 and len(val_eeg) > 0:
        print(f"\n{'═'*60}")
        for k in range(min(3, len(val_eeg))):
            gen = generate(val_eeg[k])
            ref = val_sents[k]
            ref_w = set(ref.lower().split())
            gen_w = set(gen.lower().split())
            pct   = int(100 * len(ref_w & gen_w)
                        / max(len(ref_w), 1))
            print(f"REF: {ref[:70]}")
            print(f"GEN: {gen[:70]}")
            print(f"Match: {pct}%\n")
        print(f"{'═'*60}\n")


# SAVE LOG
with open(LOG_PATH, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['epoch','train_loss',
                'val_loss','gap'])
    w.writerows(log_rows)

print(f"\nDone! Best val: {best_val_loss:.4f}")
print(f"Saved: {SAVE_PATH}")
print("Next: python Evaluate.py")