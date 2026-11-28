"""Fusione di piu' sistemi a livello di ipotesi (senza LLM esterno).

Per ogni sistema: punteggio finale = ac_scale*ac + (1-alpha)*ng + alpha*llm (pesi da rescore_*.json),
softmax sulle n-best -> posterior per frase. Somma pesata delle posterior tra sistemi, sceglie la massima.
Serve per combinare modelli i cui logits non sono allineati nel tempo (non mediabili frame per frame).

Uso: python src/nbest_vote.py SYS1 SYS2 ... [--w 1,1,...] [--temp 1.0] [--write NOME]
"""
import argparse, json, sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rescore_eval import wer, clean  # noqa: E402

ROOT = '/mnt/20TB/asaccoia/brain2text'
LM = '5gram_unpr'


def posts(run, split, temp):
    d = f'{ROOT}/runs/{run}'
    nb = json.load(open(f'{d}/nbest_{LM}_{split}.json'))
    ll = json.load(open(f'{d}/llm_opt67_{LM}_{split}.json'))
    r = json.load(open(f'{d}/rescore_{LM}_opt67.json'))
    out = []
    for hyps, ls in zip(nb, ll):
        acc = {}
        if hyps:
            s = r['ac_scale'] * np.array([h[1] for h in hyps]) + (1 - r['alpha']) * np.array([h[2] for h in hyps]) \
                + r['alpha'] * np.array(ls)
            p = np.exp((s - s.max()) / temp); p /= p.sum()
            for h, pi in zip(hyps, p):
                c = clean(h[0])
                if c:
                    acc[c] = acc.get(c, 0.0) + float(pi)
        out.append(acc)
    return out


def fuse(P, w):
    res = []
    for i in range(len(P[0])):
        tot = {}
        for pk, wk in zip(P, w):
            for c, v in pk[i].items():
                tot[c] = tot.get(c, 0.0) + wk * v
        res.append(max(tot, key=tot.get) if tot else '')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('systems', nargs='+')
    ap.add_argument('--w', default='')
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--write', default='')
    a = ap.parse_args()
    w = [float(x) for x in a.w.split(',')] if a.w else [1.0] * len(a.systems)
    meta = json.loads(str(np.load(f'{ROOT}/runs/{a.systems[0]}/logits_test.npz', allow_pickle=True)['meta']))
    refs = [m['text'] for m in meta]
    ud = sorted(set(m['day'] for m in meta)); sel = [i for i, m in enumerate(meta) if 4 <= ud.index(m['day']) <= 18]
    P = [posts(s, 'test', a.temp) for s in a.systems]
    for s, pk in zip(a.systems, P):
        h = [max(x, key=x.get) if x else '' for x in pk]
        print(f'{s:24s} WER {wer(refs, h):.4f}  sub600 {wer([refs[i] for i in sel], [h[i] for i in sel]):.4f}')
    h = fuse(P, w)
    print(f'{"FUSIONE":24s} WER {wer(refs, h):.4f}  sub600 {wer([refs[i] for i in sel], [h[i] for i in sel]):.4f}')
    if a.write:
        hc = fuse([posts(s, 'competition', a.temp) for s in a.systems], w)
        open(f'{ROOT}/submissions/{a.write}.txt', 'w').write('\n'.join(hc) + '\n')
        print('scritto', f'submissions/{a.write}.txt')


if __name__ == '__main__':
    main()
