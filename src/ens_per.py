"""PER greedy sul test di singoli run e della media delle log-prob (stessa frequenza dei frame).

Uso: python src/ens_per.py run1 run2 ... [--mode prob|logprob] [--write NOME]
--write salva i logits medi in runs/NOME/logits_{test,competition}.npz (per la decodifica n-gram).
"""
import argparse, json, os, pickle, sys
import numpy as np

ROOT = '/mnt/20TB/asaccoia/brain2text'


def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]; d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1])); prev = cur
    return d[len(b)]


def greedy(lp):
    g = lp.argmax(-1); out = []; prev = -1
    for c in g:
        if c != prev and c != 0: out.append(int(c))
        prev = c
    return out


def per(logits, refs):
    e = n = 0
    for lp, y in zip(logits, refs):
        e += ed(list(y), greedy(lp)); n += len(y)
    return e / n


def combine(lps, mode):
    L = min(x.shape[0] for x in lps)
    xs = np.stack([x[:L].astype(np.float32) for x in lps])
    if mode == 'prob':
        m = np.log(np.exp(xs).mean(0) + 1e-12)
    else:
        m = xs.mean(0); m = m - np.log(np.exp(m).sum(-1, keepdims=True))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--mode', default='prob')
    ap.add_argument('--write', default='')
    a = ap.parse_args()
    d = pickle.load(open(f'{ROOT}/data/processed/b2t24.pkl', 'rb'))
    refs = [t['phonemes'] for t in d['test']]
    L = {r: list(np.load(f'{ROOT}/runs/{r}/logits_test.npz', allow_pickle=True)['logits']) for r in a.runs}
    for r in a.runs:
        print(f'{r:28s} PER {per(L[r], refs):.4f}')
    ens = [combine([L[r][i] for r in a.runs], a.mode) for i in range(len(refs))]
    print(f'{"ENSEMBLE (" + a.mode + ")":28s} PER {per(ens, refs):.4f}  [{len(a.runs)} modelli]')
    if a.write:
        out = f'{ROOT}/runs/{a.write}'; os.makedirs(out, exist_ok=True)
        for split in ['test', 'competition']:
            zs = [np.load(f'{ROOT}/runs/{r}/logits_{split}.npz', allow_pickle=True) for r in a.runs]
            n = len(zs[0]['logits'])
            lg = [combine([z['logits'][i] for z in zs], a.mode).astype(np.float16) for i in range(n)]
            np.savez_compressed(f'{out}/logits_{split}.npz', logits=np.array(lg, dtype=object), meta=zs[0]['meta'])
        json.dump({'ensemble_of': a.runs, 'mode': a.mode}, open(f'{out}/config.json', 'w'))
        print('scritto', out)


if __name__ == '__main__':
    main()
