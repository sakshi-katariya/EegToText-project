

import torch
import torch.nn as nn
import numpy as np
from transformers import BartForConditionalGeneration, BartTokenizer


N_CHANNELS  = 105
SEQ_LEN     = 512
HIDDEN_DIM  = 512
N_LAYERS    = 3
DROPOUT     = 0.3
BART_DIM    = 768
N_PREFIX    = 8
SAVE_PATH   = 'eeg_bart_model.pt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# MODELS - identical to training.py 
class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(N_CHANNELS, 128, kernel_size=4, stride=4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=2, stride=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=HIDDEN_DIM,
            num_layers=N_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT
        )
        self.attention = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM * 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        pooled = (lstm_out * attn_weights).sum(dim=1)
        return self.norm(pooled)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(1024, BART_DIM * N_PREFIX),
            nn.LayerNorm(BART_DIM * N_PREFIX)
        )

    def forward(self, x):
        return self.net(x).view(-1, N_PREFIX, BART_DIM)


#  LOAD
print(f"Loading checkpoint from {SAVE_PATH} ...")
try:
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
except FileNotFoundError:
    print(f"ERROR: {SAVE_PATH} not found. Run train.py first!")
    exit()

print("Loading BART...")
tokenizer   = BartTokenizer.from_pretrained('facebook/bart-base')
bart        = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(DEVICE)
eeg_encoder = SpeechEncoder().to(DEVICE)
adapter     = Adapter().to(DEVICE)

for param in bart.parameters():
    param.requires_grad = False

eeg_encoder.load_state_dict(ckpt['eeg_encoder'])
adapter.load_state_dict(ckpt['adapter'])
print(f"Model loaded! (trained to epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")


#  INFERENCE 
def eeg_to_text(eeg_array, max_tokens=40, num_beams=4):
    """
    Input : numpy array (105, 512) — normalized EEG segment
    Output: decoded sentence string
    """
    assert eeg_array.shape == (N_CHANNELS, SEQ_LEN), \
        f"Wrong shape! Expected ({N_CHANNELS},{SEQ_LEN}), got {eeg_array.shape}"

    eeg_encoder.eval(); adapter.eval(); bart.eval()

    with torch.no_grad():
        eeg_t    = torch.FloatTensor(eeg_array).unsqueeze(0).to(DEVICE)
        eeg_feat = eeg_encoder(eeg_t)       # (1, 1024)
        prefix   = adapter(eeg_feat)         # (1, N_PREFIX, 768)
        prefix_mask = torch.ones(1, N_PREFIX, dtype=torch.long).to(DEVICE)

        # beam search for better quality output
        output = bart.generate(
            encoder_outputs=(prefix,),
            attention_mask=prefix_mask,
            max_new_tokens=max_tokens,
            num_beams=num_beams,           # beam search (better than greedy)
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            forced_bos_token_id=tokenizer.bos_token_id
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


#  TEST 
print("\n--- TEST WITH RANDOM EEG ---")
print("(Replace with real EEG for actual results)\n")

fake_eeg = np.random.randn(N_CHANNELS, SEQ_LEN).astype(np.float32)
for i in range(3):
    result = eeg_to_text(fake_eeg)
    print(f"Sample {i+1}: {result}")

print("\nDONE! Run evaluate.py for ROUGE-L scoring.")
