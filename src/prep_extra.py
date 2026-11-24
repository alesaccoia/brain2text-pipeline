"""Estrae dai file data/extracted/sentences i blocchi esclusi da competitionData (dati di addestramento in piu').

Prende solo i blocchi "Chang" (frasi del vocabolario da 50 parole), che competitionData esclude per
costruzione. I blocchi "Switchboard Repeat" sono scartati: potrebbero ripetere frasi della partizione
di gara. Scarta anche ogni frase il cui testo compare nel test.
Ogni trial = periodo "go" (goTrialEpochs), z-scored per blocco come in prep.py.
Scrive data/processed/b2t24_extra.pkl: {'extra': [trial, ...]} con lo stesso formato di prep.py.
"""
import os, pickle, re, sys
import numpy as np
import scipy.io

sys.path.insert(0, os.path.dirname(__file__))
from prep import clean_text, to_phonemes, ROOT  # noqa: E402

SENT = f'{ROOT}/data/extracted/sentences/sentences'
COMP = f'{ROOT}/data/extracted/competitionData/competitionData'
FEATS = ['tx1', 'tx2', 'tx3', 'tx4', 'spikePow']


def main():
    base = pickle.load(open(f'{ROOT}/data/processed/b2t24.pkl', 'rb'))
    sessions = base['sessions']
    test_txt = {t['text'] for t in base['test']}
    extra, skipped = [], 0
    for di, s in enumerate(sessions):
        m = scipy.io.loadmat(f'{SENT}/{s}_sentences.mat')
        used = set()
        for part in ['train', 'test', 'competitionHoldOut']:
            p = f'{COMP}/{part}/{s}.mat'
            if os.path.exists(p):
                used |= set(np.unique(scipy.io.loadmat(p, variable_names=['blockIdx'])['blockIdx']).tolist())
        bl = [int(b) for b in np.squeeze(m['blockList'])]
        bt = [str(x[0]) for x in np.squeeze(m['blockTypes'])]
        keep = {b for b, t in zip(bl, bt) if b not in used and 'Chang' in t}
        if not keep:
            continue
        bn = np.squeeze(m['blockNum'])
        ge = m['goTrialEpochs']
        F = np.concatenate([m[k] for k in FEATS], axis=1).astype(np.float32)
        trials = []
        for k, (a, b) in enumerate(ge):
            blk = int(bn[a])
            if blk not in keep:
                continue
            raw = str(m['sentences'][k][0][0]).strip() if m['sentences'][k][0].size else ''
            txt = clean_text(raw)
            if not txt or txt in test_txt:
                skipped += 1
                continue
            trials.append(dict(x=F[a:b], day=di, block=blk, partition='extra', order=k, raw_text=raw, text=txt))
        for blk in sorted({t['block'] for t in trials}):
            idx = [i for i, t in enumerate(trials) if t['block'] == blk]
            allf = np.concatenate([trials[i]['x'] for i in idx], axis=0)
            mu, sd = allf.mean(0, keepdims=True), allf.std(0, keepdims=True)
            for i in idx:
                trials[i]['x'] = ((trials[i]['x'] - mu) / (sd + 1e-8)).astype(np.float16)
        for t in trials:
            t['phonemes'] = np.array(to_phonemes(t['text']), dtype=np.int64)
        extra += trials
        print(s, 'blocchi', sorted(keep), 'trial', len(trials), flush=True)
    pickle.dump({'extra': extra}, open(f'{ROOT}/data/processed/b2t24_extra.pkl', 'wb'), protocol=4)
    print(f'salvati {len(extra)} trial extra ({skipped} scartati)')


if __name__ == '__main__':
    main()
