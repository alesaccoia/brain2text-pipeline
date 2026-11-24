"""Converte competitionData (.mat) in un pickle per l'addestramento.

Per ogni trial salva tutte le feature (tx1..tx4 + spikePow, 256 canali ciascuno = 1280 colonne),
z-scored per blocco come nella baseline, in float16. Le colonne sono in ordine
[tx1 | tx2 | tx3 | tx4 | spikePow]; i canali 0:128 di ogni gruppo sono l'area 6v.
"""
import os, re, pickle, sys
import numpy as np
import scipy.io
from g2p_en import G2p

ROOT = '/mnt/20TB/asaccoia/brain2text'
DATA = f'{ROOT}/data/extracted/competitionData/competitionData'
OUT = f'{ROOT}/data/processed/b2t24.pkl'

PHONE_DEF = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
             'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
             'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
PHONE_DEF_SIL = PHONE_DEF + ['SIL']  # id CTC = indice+1, 0 = blank

g2p = G2p()


def clean_text(s):
    s = re.sub(r"[^a-zA-Z\- ']", '', s.strip())
    s = s.replace('--', '').lower()
    return s


def to_phonemes(text):
    ph = []
    for p in g2p(text):
        if p == ' ':
            ph.append('SIL')
        p = re.sub(r'[0-9]', '', p)
        if re.match(r'[A-Z]+', p):
            ph.append(p)
    ph.append('SIL')
    # rimuovi SIL doppi
    out = []
    for p in ph:
        if p == 'SIL' and out and out[-1] == 'SIL':
            continue
        out.append(p)
    return [PHONE_DEF_SIL.index(p) + 1 for p in out]


def load_partition(path, day_idx, partition):
    d = scipy.io.loadmat(path)
    n = d['sentenceText'].shape[0]
    feats = []
    for i in range(n):
        f = np.concatenate([d[k][0, i] for k in ['tx1', 'tx2', 'tx3', 'tx4', 'spikePow']], axis=1).astype(np.float32)
        feats.append(f)
    blocks = np.squeeze(d['blockIdx'])
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        allf = np.concatenate([feats[i] for i in idx], axis=0)
        mu, sd = allf.mean(0, keepdims=True), allf.std(0, keepdims=True)
        for i in idx:
            feats[i] = ((feats[i] - mu) / (sd + 1e-8)).astype(np.float16)
    trials = []
    for i in range(n):
        raw = str(d['sentenceText'][i]).strip()
        t = dict(x=feats[i], day=day_idx, block=int(blocks[i]), partition=partition, order=i, raw_text=raw)
        if partition != 'competition':
            t['text'] = clean_text(raw)
            t['phonemes'] = np.array(to_phonemes(t['text']), dtype=np.int64)
        trials.append(t)
    return trials


def main():
    sessions = sorted(f[:-4] for f in os.listdir(f'{DATA}/train'))
    out = {'sessions': sessions, 'phone_def': PHONE_DEF_SIL, 'train': [], 'test': [], 'competition': []}
    for di, s in enumerate(sessions):
        out['train'] += load_partition(f'{DATA}/train/{s}.mat', di, 'train')
        out['test'] += load_partition(f'{DATA}/test/{s}.mat', di, 'test')
        p = f'{DATA}/competitionHoldOut/{s}.mat'
        if os.path.exists(p):
            out['competition'] += load_partition(p, di, 'competition')
        print(s, len(out['train']), len(out['test']), len(out['competition']), flush=True)
    with open(OUT, 'wb') as fh:
        pickle.dump(out, fh, protocol=4)
    print('salvato', OUT)


if __name__ == '__main__':
    main()
