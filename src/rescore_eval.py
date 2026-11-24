"""Combina punteggi acustici, n-gram e LLM; tara i pesi sul test e produce la submission.

score = acoustic_scale * ac + (1 - alpha) * ngram_lm + alpha * llm
Uso: python src/rescore_eval.py --run NOME --lm 5gram --llm opt67 [--write_submission]
Stampa il WER sul test (1-best n-gram e con rescoring) e salva runs/NOME/rescore_{lm}_{llm}.json.
"""
import argparse, json, os
import numpy as np
import jiwer

ROOT = '/mnt/20TB/asaccoia/brain2text'


def wer(refs, hyps):
    e = n = 0
    for r, h in zip(refs, hyps):
        o = jiwer.process_words(r, h if h.strip() else '<empty>')
        e += o.substitutions + o.deletions + o.insertions
        n += len(r.split())
    return e / n


def pick(nb, llm, ac_scale, alpha):
    out = []
    for hyps, ls in zip(nb, llm):
        if not hyps:
            out.append('')
            continue
        ac = np.array([h[1] for h in hyps]); ng = np.array([h[2] for h in hyps]); ll = np.array(ls)
        out.append(hyps[int(np.argmax(ac_scale * ac + (1 - alpha) * ng + alpha * ll))][0])
    return out


def clean(s):
    return ' '.join(s.lower().replace('.', '').replace(',', '').replace('?', '').split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--lm', default='5gram')
    ap.add_argument('--llm', default='opt67')
    ap.add_argument('--write_submission', action='store_true')
    a = ap.parse_args()
    d = f'{ROOT}/runs/{a.run}'
    meta = json.loads(str(np.load(f'{d}/logits_test.npz', allow_pickle=True)['meta']))
    refs = [m['text'] for m in meta]
    nb = json.load(open(f'{d}/nbest_{a.lm}_test.json'))
    llm = json.load(open(f'{d}/llm_{a.llm}_{a.lm}_test.json'))
    base = wer(refs, [clean(h[0][0]) if h else '' for h in nb])
    best = (9, None)
    for acs in [0.3, 0.5, 0.7, 1.0, 1.5]:
        for al in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            w = wer(refs, [clean(s) for s in pick(nb, llm, acs, al)])
            if w < best[0]:
                best = (w, (acs, al))
    print(f'{a.run}: WER test 1-best {a.lm} = {base:.4f} | + {a.llm} = {best[0]:.4f} con (ac_scale, alpha) = {best[1]}')
    res = {'wer_ngram': base, 'wer_rescored': best[0], 'ac_scale': best[1][0], 'alpha': best[1][1]}
    json.dump(res, open(f'{d}/rescore_{a.lm}_{a.llm}.json', 'w'))
    if a.write_submission:
        nbc = json.load(open(f'{d}/nbest_{a.lm}_competition.json'))
        llc = json.load(open(f'{d}/llm_{a.llm}_{a.lm}_competition.json'))
        hyps = [clean(s) for s in pick(nbc, llc, *best[1])]
        assert len(hyps) == 1200
        p = f'{ROOT}/submissions/{a.run}_{a.lm}_{a.llm}.txt'
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').write('\n'.join(hyps) + '\n')
        print('submission scritta in', p)


if __name__ == '__main__':
    main()
