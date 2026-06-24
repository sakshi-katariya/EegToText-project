

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


N_CHANNELS = 105
SEQ_LEN    = 512
PATCH_SIZE = 16
D_MODEL    = 256
N_PREFIX   = 4
SAVE_PATH  = 'eeg_word_model.pt'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


#  MODELS - must be identical to gpt2_connects.py 
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


# LOAD CHECKPOINT
print(f"Loading checkpoint from {SAVE_PATH} ...")
try:
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE)
except FileNotFoundError:
    print(f"ERROR: {SAVE_PATH} not found.")
    print("Run gpt2_connects.py first to train the model.")
    exit()

# auto-detect which LLM was used during training
LLM_NAME = ckpt.get('llm_name', 'gpt2-large')
EMB_DIM  = ckpt.get('emb_dim',  1280)
print(f"LLM            : {LLM_NAME}")
print(f"Embedding dim  : {EMB_DIM}")

# LOAD LLM 
print(f"Loading {LLM_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
tokenizer.pad_token = tokenizer.eos_token
llm = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)

for param in llm.parameters():
    param.requires_grad = False

# LOAD ENCODER + ADAPTER
encoder = EEGEncoder().to(DEVICE)
adapter = Adapter(out_dim=EMB_DIM).to(DEVICE)
encoder.load_state_dict(ckpt['encoder'])
adapter.load_state_dict(ckpt['adapter'])
print("✓ All weights loaded successfully!")


#  INFERENCE 
def eeg_to_text(eeg_array, max_tokens=30):
    """
    Input : numpy array of shape (105, 512)
            — one EEG segment, normalized per channel
    Output: decoded sentence string
    """
    assert eeg_array.shape == (N_CHANNELS, SEQ_LEN), \
        f"Wrong shape! Expected ({N_CHANNELS},{SEQ_LEN}), got {eeg_array.shape}"

    encoder.eval(); adapter.eval(); llm.eval()

    with torch.no_grad():
        eeg_t  = torch.FloatTensor(eeg_array).unsqueeze(0).to(DEVICE)  # (1,105,512)
        prefix = adapter(encoder(eeg_t))                                 # (1,4,EMB_DIM)
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


# TEST 
print("\n--- TEST WITH RANDOM EEG ---")
print("(Replace fake_eeg with real EEG chunk for actual results)\n")

fake_eeg = np.random.randn(N_CHANNELS, SEQ_LEN).astype(np.float32)
for i in range(3):
    result = eeg_to_text(fake_eeg)
    print(f"Sample {i+1}: {result}")

print("\nDONE!")
print("Next step: run evaluate.py to get ROUGE-L score")
