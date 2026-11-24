"""Decodifica WFST n-gram dei logits salvati da train.py (gira in .venv-lm, che ha lm_decoder).

Modalita' "servizio": carica il LM una volta, poi scorre per sempre runs/*/logits_{test,competition}.npz
e scrive runs/RUN/nbest_{LM}_{split}.json (lista per frase di [frase, ac_score, lm_score]).
Stampa il WER 1-best sul test.

Uso: python src/decode_ngram.py --lm 5gram [--once] [--runs nome1,nome2] [--shard i/n]
"""
import argparse, glob, json, os, time, zlib
import numpy as np
import lm_decoder

ROOT = '/mnt/20TB/asaccoia/brain2text'
LMS = {
    '3gram': f'{ROOT}/data/extracted/languageModel/languageModel',
    '5gram': f'{ROOT}/data/extracted/languageModel_5gram/speech_5gram/lang_test',
}
# ordine kaldi: [blank, SIL, AA..ZH]; ordine nostro: [blank, AA..ZH (1..39), SIL (40)]
KALDI_ORDER = [0, 40] + list(range(1, 40))


def build(lm, acoustic_scale, nbest, beam, unpruned_rescore):
    p = LMS[lm]
    opts = lm_decoder.DecodeOptions(7000, 200, beam, 8.0, acoustic_scale, 1.0, 0.0, nbest)
    g = os.path.join(p, 'G.fst') if unpruned_rescore else ''
    gr = os.path.join(p, 'G_no_prune.fst') if unpruned_rescore else ''
    res = lm_decoder.DecodeResource(os.path.join(p, 'TLG.fst'), g, gr, os.path.join(p, 'words.txt'), '')
    return lm_decoder.BrainSpeechDecoder(res, opts)


def decode_one(dec, lp, blank_penalty, rescore):
    lp = lp.astype(np.float32)[:, KALDI_ORDER]
    lm_decoder.DecodeNumpy(dec, lp, np.zeros([1, lp.shape[1]], dtype=np.float32), blank_penalty)
    dec.FinishDecoding()
    if rescore:
        dec.Rescore()
    out = [[r.sentence.strip(), float(r.ac_score), float(r.lm_score)] for r in dec.result()]
    dec.Reset()
    return out


def wer(refs, hyps):
    import jiwer
    errs = sum(jiwer.process_words(r, h if h else '<empty>').wer * len(r.split()) for r, h in zip(refs, hyps))
    return errs / sum(len(r.split()) for r in refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lm', default='5gram')
    ap.add_argument('--acoustic_scale', type=float, default=0.5)
    ap.add_argument('--blank_penalty', type=float, default=float(np.log(7)))
    ap.add_argument('--nbest', type=int, default=100)
    ap.add_argument('--beam', type=float, default=18.0)
    ap.add_argument('--unpruned', type=int, default=0)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--runs', default='')
    ap.add_argument('--shard', default='0/1')
    a = ap.parse_args()
    si, sn = map(int, a.shard.split('/'))
    t0 = time.time()
    dec = build(a.lm, a.acoustic_scale, a.nbest, a.beam, a.unpruned)
    print(f'LM {a.lm} caricato in {time.time()-t0:.0f}s', flush=True)
    tag = a.lm + ('_unpr' if a.unpruned else '')
    while True:
        todo = []
        for f in sorted(glob.glob(f'{ROOT}/runs/*/logits_*.npz')):
            run = f.split('/')[-2]
            if a.runs and run not in a.runs.split(','):
                continue
            if sn > 1 and zlib.crc32(run.encode()) % sn != si:
                continue
            split = f.split('logits_')[-1][:-4]
            if split == 'train':
                continue
            out = f'{ROOT}/runs/{run}/nbest_{tag}_{split}.json'
            if not os.path.exists(out):
                todo.append((run, split, f, out))
        for run, split, f, out in todo:
            z = np.load(f, allow_pickle=True)
            logits, meta = z['logits'], json.loads(str(z['meta']))
            t1 = time.time()
            res = [decode_one(dec, lp, a.blank_penalty, bool(a.unpruned)) for lp in logits]
            json.dump(res, open(out + '.tmp', 'w'))
            os.replace(out + '.tmp', out)
            msg = f'{run} {split}: {len(res)} frasi in {time.time()-t1:.0f}s'
            if split == 'test':
                refs = [m['text'] for m in meta]
                hyps = [r[0][0] if r else '' for r in res]
                w = wer(refs, hyps)
                msg += f'  WER 1-best {tag} = {w:.4f}'
                json.dump({'wer_1best': w}, open(f'{ROOT}/runs/{run}/wer_{tag}_test.json', 'w'))
            print(time.strftime('%H:%M:%S ') + msg, flush=True)
        if a.once:
            break
        time.sleep(120)


if __name__ == '__main__':
    main()
