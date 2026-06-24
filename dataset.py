import os
import numpy as np
import scipy.io
import pandas as pd

CSV_FOLDER = '/home/ubuntu/eeg/'
EEG_FOLDER = '/home/ubuntu/raw_eeg/'
N_CHANNELS = 105
SEQ_LEN    = 512


def load_all_data():
    all_eeg       = []
    all_sentences = []

    for i in range(1, 8):
        csv_path = os.path.join(CSV_FOLDER, f'nr_{i}.csv')
        eeg_path = os.path.join(EEG_FOLDER, f'YAC_NR{i}_ET.mat')

        # skip if either file is missing
        if not os.path.exists(csv_path):
            print(f"[SKIP] CSV missing : {csv_path}")
            continue
        if not os.path.exists(eeg_path):
            print(f"[SKIP] EEG missing : {eeg_path}")
            continue

        # Load sentences 
        df        = pd.read_csv(csv_path, sep=';', header=None)
        sentences = df.iloc[:, 2].dropna().unique().tolist()
        print(f"NR{i} → {len(sentences)} sentences loaded")

        # Load EEG 
        mat     = scipy.io.loadmat(eeg_path)
        eeg_raw = mat['data'].astype(np.float32)

        # make sure shape is (channels, time)
        if eeg_raw.shape[0] != N_CHANNELS:
            eeg_raw = eeg_raw.T

        assert eeg_raw.shape[0] == N_CHANNELS, \
            f"ERROR: Expected {N_CHANNELS} channels, got {eeg_raw.shape[0]}"

        print(f"NR{i} → EEG shape : {eeg_raw.shape}")

        # Normalize per channel
        mean    = eeg_raw.mean(axis=1, keepdims=True)
        std     = eeg_raw.std(axis=1,  keepdims=True) + 1e-8
        eeg_raw = (eeg_raw - mean) / std

        #  Chunk into (105, 512) segments 
        n_chunks = min(eeg_raw.shape[1] // SEQ_LEN, len(sentences))
        for j in range(n_chunks):
            chunk = eeg_raw[:, j*SEQ_LEN:(j+1)*SEQ_LEN]  # (105, 512)
            all_eeg.append(chunk.astype(np.float32))
            all_sentences.append(str(sentences[j]))

        print(f"NR{i} → {n_chunks} EEG-sentence pairs created\n")

    # summary
    print(f"{'─'*45}")
    print(f"Total EEG chunks  : {len(all_eeg)}")
    print(f"Total sentences   : {len(all_sentences)}")
    print(f"Unique sentences  : {len(set(all_sentences))}")
    if len(all_eeg) > 0:
        print(f"Single chunk shape: {all_eeg[0].shape}")
        print(f"Sample sentence   : {all_sentences[0]}")
        print(f"EEG mean          : {all_eeg[0].mean():.4f}")
        print(f"EEG std           : {all_eeg[0].std():.4f}")
    print(f"{'─'*45}")

    return all_eeg, all_sentences


if __name__ == '__main__':
    eeg_list, sent_list = load_all_data()
    if len(eeg_list) == 0:
        print("\nERROR: No data loaded. Check your file paths above.")
    else:
        print("\nData check PASSED — ready to run gpt2_connects.py")