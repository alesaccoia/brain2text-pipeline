"""Fusione finale con un LLM (via proxy OpenAI-compatibile su 127.0.0.1:8080).

Per ogni frase raccoglie le migliori ipotesi (dopo rescoring 5-gram + OPT) di piu' modelli,
le mostra all'LLM con qualche esempio risolto e chiede la trascrizione piu' probabile.

Uso:
  python src/llm_merge.py --split test --n 100          # prova: WER su 100 frasi di test
  python src/llm_merge.py --split competition           # scrive la submission
Gli esempi nel prompt vengono dal test (frasi diverse da quelle valutate).
"""
import argparse, json, os, random, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rescore_eval import wer, clean

ROOT = '/mnt/20TB/asaccoia/brain2text'
RUNS = ['tf_d512_m50_s0', 'tf_d384_m50_s1', 'tf_d384_m50_s2', 'tf_d384_m50_40k_s0', 'tf_d384_m50_s0', 'gru_base_s0']
LM = '5gram_unpr'
URL = 'http://127.0.0.1:8080/v1/chat/completions'
KEY = os.environ.get('PROXY_KEY') or (open(f'{ROOT}/.proxy_key').read().strip() if os.path.exists(f'{ROOT}/.proxy_key') else '')

SYSTEM = (
    "You are the final stage of a brain-to-text speech decoder. A person with ALS attempted to speak an "
    "English sentence (conversational, Switchboard-like). Several neural decoders + language models produced "
    "candidate transcriptions; each candidate is listed with the number of decoders that ranked it highly and "
    "its best rank. Candidates usually share most words; errors are mostly phonetically similar words, "
    "dropped short words, or wrong word boundaries. Output the single most likely true sentence: lowercase, "
    "no punctuation except apostrophes. You may combine parts of different candidates or fix an obvious "
    "word, but stay close to the candidates. Reply with the sentence only."
)


SYSTEM2 = (
    "You are the final stage of a brain-to-text speech decoder. A person with ALS attempted to speak an "
    "English sentence (conversational, Switchboard-like). Several neural decoders produced phoneme probabilities; "
    "each was decoded with a 5-gram language model and rescored with an LLM. For each candidate you get its "
    "average posterior across decoders (higher = more confident), how many decoders proposed it, and its best "
    "rank. You also get the raw greedy phoneme sequence (CMU ARPAbet) of the best decoder, read directly from "
    "the neural signal before any language model: use it to check which words were actually attempted, and "
    "to recover words that the language models replaced with more common ones. Errors are mostly phonetically "
    "similar words, dropped short words, or wrong word boundaries. Output the single most likely true sentence: "
    "lowercase, no punctuation except apostrophes. Stay close to the candidates and phonemes. "
    "Reply with the sentence only."
)


def ranked(run, split, k):
    d = f'{ROOT}/runs/{run}'
    nb = json.load(open(f'{d}/nbest_{LM}_{split}.json'))
    ll = json.load(open(f'{d}/llm_opt67_{LM}_{split}.json'))
    r = json.load(open(f'{d}/rescore_{LM}_opt67.json'))
    out = []
    for hyps, ls in zip(nb, ll):
        if not hyps:
            out.append([]); continue
        ac = np.array([h[1] for h in hyps]); ng = np.array([h[2] for h in hyps])
        s = r['ac_scale'] * ac + (1 - r['alpha']) * ng + r['alpha'] * np.array(ls)
        post = np.exp(s - s.max()); post /= post.sum()
        acc = {}
        for i in np.argsort(-s):
            c = clean(hyps[i][0])
            if c:
                acc[c] = acc.get(c, 0.0) + float(post[i])
        lst = sorted(acc.items(), key=lambda x: -x[1])[:k]
        out.append(lst if V2 else [c for c, _ in lst])
    return out


PHONE_DEF = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G',
             'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
             'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
PHONE_DEF_SIL = PHONE_DEF + ['SIL']  # id CTC = indice+1, 0 = blank

V2 = False
PHONES = None


def greedy_phones(split):
    z = np.load(f'{ROOT}/runs/{RUNS[0]}/logits_{split}.npz', allow_pickle=True)['logits']
    out = []
    for lp in z:
        ids = lp.argmax(-1); seq, prev = [], -1
        for t in ids:
            if t != prev and t != 0:
                seq.append(PHONE_DEF_SIL[t - 1])
            prev = t
        out.append(' '.join(seq).replace('SIL', '|').strip(' |'))
    return out


def candidates(per_run, i):
    agg = {}
    for lists in per_run:
        for rank, c in enumerate(lists[i]):
            if V2:
                c, p = c
            else:
                p = 0.0
            v = agg.setdefault(c, [0, 99, 0.0])
            v[0] += 1; v[1] = min(v[1], rank + 1); v[2] += p / len(per_run)
    if not V2:
        items = sorted(agg.items(), key=lambda x: (-x[1][0], x[1][1]))[:12]
        return '\n'.join(f'- "{c}" (decoders: {v[0]}, best rank: {v[1]})' for c, v in items), items[0][0]
    items = sorted(agg.items(), key=lambda x: -x[1][2])[:12]
    txt = '\n'.join(f'- "{c}" (avg posterior: {v[2]:.3f}, decoders: {v[0]}/{len(per_run)}, best rank: {v[1]})' for c, v in items)
    txt = f'Greedy phonemes from best decoder (| = word boundary, may contain errors): {PHONES[i]}\n' + txt
    return txt, items[0][0]


def call(messages, model, effort, tries=10):
    body = json.dumps({'model': model, 'messages': messages, 'reasoning_effort': effort}).encode()
    for t in range(tries):
        try:
            req = urllib.request.Request(URL, body, {'Content-Type': 'application/json', 'Authorization': f'Bearer {KEY}'})
            r = json.load(urllib.request.urlopen(req, timeout=300))
            return r['choices'][0]['message']['content'], r.get('usage', {})
        except Exception as e:
            err = e
            time.sleep(10 * (t + 1))
    raise err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--n', type=int, default=0)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--shots', type=int, default=40)
    ap.add_argument('--model', default='gpt-6.2-astra')
    ap.add_argument('--effort', default='medium')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tag', default='astra')
    ap.add_argument('--v2', type=int, default=0)
    a = ap.parse_args()
    global V2, PHONES
    V2 = bool(a.v2)

    meta_t = json.loads(str(np.load(f'{ROOT}/runs/{RUNS[0]}/logits_test.npz', allow_pickle=True)['meta']))
    refs_t = [m['text'] for m in meta_t]
    pr_t = [ranked(r, 'test', a.k) for r in RUNS]
    phones_t = greedy_phones('test')
    PHONES = phones_t
    rng = random.Random(0)
    idx = list(range(len(refs_t))); rng.shuffle(idx)
    shot_idx = idx[:a.shots]
    if a.split == 'test':
        pr, refs = pr_t, refs_t
        ev = idx[a.shots:]
        if a.n: ev = ev[:a.n]
    else:
        pr = [ranked(r, a.split, a.k) for r in RUNS]
        refs = None
        phones_c = greedy_phones(a.split)
        ev = list(range(len(pr[0])))
        shot_idx = idx[:a.shots]

    shots = '\n\n'.join(f'Candidates:\n{candidates(pr_t, j)[0]}\nTrue sentence: {clean(refs_t[j])}' for j in shot_idx)
    sysmsg = (SYSTEM2 if V2 else SYSTEM) + '\n\nSolved examples:\n\n' + shots
    if a.split != 'test':
        PHONES = phones_c

    def one(i):
        cand, top = candidates(pr, i)
        try:
            txt, u = call([{'role': 'system', 'content': sysmsg},
                           {'role': 'user', 'content': f'Candidates:\n{cand}\nTrue sentence:'}], a.model, a.effort)
            txt = clean(re.sub(r'^true sentence:\s*', '', txt.strip().strip('"').lower()))
            return i, txt or top, top, u
        except Exception as e:
            print('errore', i, e, flush=True)
            return i, top, top, {}

    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, ev))
    tok = sum(r[3].get('total_tokens', 0) for r in res)
    out = {i: h for i, h, _, _ in res}
    print(f'{len(res)} frasi in {time.time()-t0:.0f}s, token totali {tok}', flush=True)
    if a.split == 'test':
        json.dump({'ev': ev, 'hyp': [out[i] for i in ev]}, open(f'{ROOT}/runs/merge_{a.tag}_test.json', 'w'))
        R = [refs[i] for i in ev]
        best1 = [(pr[0][i][0][0] if V2 else pr[0][i][0]) if pr[0][i] else '' for i in ev]
        vote = [t for _, _, t, _ in res]
        print(f'WER {RUNS[0]} da solo: {wer(R, best1):.4f} | voto tra modelli: {wer(R, vote):.4f} | {a.model}: {wer(R, [out[i] for i in ev]):.4f}')
        json.dump({'ev': ev, 'hyp': [out[i] for i in ev]}, open(f'{ROOT}/runs/merge_{a.tag}_test.json', 'w'))
    else:
        p = f'{ROOT}/submissions/merge_{a.tag}_{"+".join(r.split("_")[1] + r.split("_")[-1] for r in RUNS)}.txt'
        open(p, 'w').write('\n'.join(out[i] for i in ev) + '\n')
        print('submission scritta in', p)


if __name__ == '__main__':
    main()
