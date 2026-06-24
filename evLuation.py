import os
import numpy as np
import scipy.io
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer


CSV_FOLDER = '/home/ubuntu/eeg/'
EEG_FOLDER = '/home/ubuntu/raw_eeg/'
SAVE_PATH  = 'eeg_word_model.pt'
LOG_PATH   = 'training_log.csv'

N_CHANNELS = 105
SEQ_LEN    = 512
PATCH_SIZE = 16
D_MODEL    = 256
N_PREFIX   = 4
N_EVAL     = 20   

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


#  MODELS  - must be identical to gpt2_connects.py 
class EEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = PATCH_SIZE
        self.n_patches  = SEQ_LEN // PATCH_SIZE
        patch_dim       = N_CHANNELS * PATCH_SIZE

        self.proj      = nn.Linear(patch_dim, D_MODEL)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches, D_MODEL) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=8,
            dim_feedforward=1024,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=6)
        self.norm    = nn.LayerNorm(D_MODEL)

    def forward(self, x):
        B, C, T = x.shape
        x = x.reshape(B, C, self.n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, self.n_patches, -1)
        x = self.proj(x) + self.pos_embed
        return self.norm(self.encoder(x)).mean(dim=1)


class Adapter(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(D_MODEL, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024,    1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024,    out_dim * N_PREFIX),
            nn.LayerNorm(out_dim * N_PREFIX)
        )

    def forward(self, x):
        return self.net(x).view(-1, N_PREFIX, self.out_dim)


#  LOAD CHECKPOINT 
print(f"Loading checkpoint from {SAVE_PATH} ...")
try:
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
except FileNotFoundError:
    print(f"ERROR: {SAVE_PATH} not found. Run gpt2_connects.py first!")
    exit()
except Exception as e:
    print(f"ERROR loading checkpoint: {e}")
    exit()

LLM_NAME = ckpt.get('llm_name', 'gpt2-large')
EMB_DIM  = ckpt.get('emb_dim',  1280)
print(f"LLM           : {LLM_NAME}")
print(f"Embedding dim : {EMB_DIM}")


# LOAD dataset
print(f"Loading {LLM_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)

for param in llm.parameters():
    param.requires_grad = False

encoder = EEGEncoder().to(DEVICE)
adapter = Adapter(out_dim=EMB_DIM).to(DEVICE)
encoder.load_state_dict(ckpt['encoder'])
adapter.load_state_dict(ckpt['adapter'])
print("✓ All weights loaded!")


#  GENERATE TEXT
def eeg_to_text(eeg_array, max_tokens=30):
    encoder.eval(); adapter.eval(); llm.eval()
    with torch.no_grad():
        eeg_t  = torch.FloatTensor(eeg_array).unsqueeze(0).to(DEVICE)
        prefix = adapter(encoder(eeg_t))
        mask   = torch.ones(1, N_PREFIX, dtype=torch.long).to(DEVICE)
        output = llm.generate(
            inputs_embeds=prefix,
            attention_mask=mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


#  LOAD EVAL DATA 
mat_path = os.path.join(EEG_FOLDER, 'YAC_NR1_ET.mat')
csv_path = os.path.join(CSV_FOLDER, 'nr_1.csv')

if not os.path.exists(mat_path):
    print(f"ERROR: EEG file not found: {mat_path}")
    exit()
if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found: {csv_path}")
    exit()

mat     = scipy.io.loadmat(mat_path)
eeg_raw = mat['data'].astype(np.float32)
if eeg_raw.shape[0] != N_CHANNELS:
    eeg_raw = eeg_raw.T

mean    = eeg_raw.mean(axis=1, keepdims=True)
std     = eeg_raw.std(axis=1,  keepdims=True) + 1e-8
eeg_raw = (eeg_raw - mean) / std

df        = pd.read_csv(csv_path, sep=';', header=None)
sentences = df.iloc[:, 2].dropna().unique().tolist()

data_list = []
n = min(N_EVAL, eeg_raw.shape[1] // SEQ_LEN, len(sentences))
for i in range(n):
    chunk = eeg_raw[:, i*SEQ_LEN:(i+1)*SEQ_LEN]   # (105, 512)
    data_list.append({
        'eeg'  : chunk.astype(np.float32),
        'label': str(sentences[i])
    })

print(f"\nEvaluating {len(data_list)} samples ...\n")
print(f"{'─'*55}")


# EVALUATE 
rouge  = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scores = []

for i, sample in enumerate(data_list):
    generated = eeg_to_text(sample['eeg'])
    reference = sample['label']
    score     = rouge.score(reference, generated)['rougeL'].fmeasure
    scores.append(score)

    print(f"[{i+1:2d}] REF     : {reference}")
    print(f"      GEN     : {generated}")
    print(f"      ROUGE-L : {score:.4f}\n")

avg = float(np.mean(scores))
print(f"{'─'*55}")
print(f"LLM USED          : {LLM_NAME}")
print(f"SAMPLES EVALUATED : {len(scores)}")
print(f"AVERAGE ROUGE-L   : {avg:.4f}")
print(f"{'─'*55}")

# score guide
print("\nScore Guide:")
print("  0.0 - 0.1  → model learned nothing")
print("  0.1 - 0.2  → very early stage")
print("  0.2 - 0.3  → okay for EEG research")
print("  0.3 - 0.4  → good result")
print("  0.4+       → excellent")


# SHOW TRAINING LOG 
try:
    log = pd.read_csv(LOG_PATH)
    print(f"\nLast 5 training epochs:")
    print(log.tail(5).to_string(index=False))
except Exception:
    print("\n(training_log.csv not found — skipping)")


if __name__ == '__main__':
    print("\nEvaluation complete!")