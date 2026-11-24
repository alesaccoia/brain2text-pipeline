"""Calcola il log-prob LLM di ogni ipotesi nelle N-best (runs/RUN/nbest_{LM}_{split}.json).

Scrive runs/RUN/llm_{TAG}_{LM}_{split}.json: per frase, lista di log-prob (stesso ordine delle ipotesi).
Modalita' servizio: scorre per sempre i file nbest nuovi.

Uso: python src/llm_score.py --model facebook/opt-6.7b --tag opt67 [--load 8bit|fp16|bf16cpu] [--once]
"""
import argparse, glob, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = '/mnt/20TB/asaccoia/brain2text'


def fmt(s):
    s = s.strip()
    return (s[:1].upper() + s[1:] + '.') if s else s


@torch.no_grad()
def score(model, tok, sents, device, bs):
    out = []
    for i in range(0, len(sents), bs):
        batch = [fmt(s) for s in sents[i:i + bs]]
        enc = tok([tok.bos_token + s if tok.bos_token else s for s in batch], return_tensors='pt', padding=True,
                  add_special_tokens=False).to(device)
        logits = model(**enc).logits.float()
        lp = logits[:, :-1].log_softmax(-1)
        tgt = enc['input_ids'][:, 1:]
        m = enc['attention_mask'][:, 1:].float()
        tok_lp = lp.gather(-1, tgt[..., None])[..., 0] * m
        out += tok_lp.sum(1).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='facebook/opt-6.7b')
    ap.add_argument('--tag', default='opt67')
    ap.add_argument('--load', default='8bit', choices=['8bit', '4bit', 'fp16', 'bf16cpu'])
    ap.add_argument('--bs', type=int, default=50)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--runs', default='')
    a = ap.parse_args()
    kw = dict(cache_dir=f'{ROOT}/hf_cache')
    tok = AutoTokenizer.from_pretrained(a.model, **kw)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if a.load == 'bf16cpu':
        torch.set_num_threads(32)
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, **kw)
        device = 'cpu'
    else:
        from transformers import BitsAndBytesConfig
        q = None
        if a.load == '8bit':
            q = BitsAndBytesConfig(load_in_8bit=True)
        elif a.load == '4bit':
            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type='nf4')
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, quantization_config=q,
                                                     device_map={'': 0}, **kw)
        device = 'cuda'
    model.eval()
    print('modello caricato', a.model, flush=True)
    while True:
        for f in sorted(glob.glob(f'{ROOT}/runs/*/nbest_*.json')):
            run = f.split('/')[-2]
            if a.runs and run not in a.runs.split(','):
                continue
            base = os.path.basename(f)[len('nbest_'):-5]  # LM_split
            out = f'{ROOT}/runs/{run}/llm_{a.tag}_{base}.json'
            if os.path.exists(out):
                continue
            nb = json.load(open(f))
            t0 = time.time()
            res = [score(model, tok, [h[0] for h in hyps], device, a.bs) if hyps else [] for hyps in nb]
            json.dump(res, open(out + '.tmp', 'w'))
            os.replace(out + '.tmp', out)
            print(time.strftime('%H:%M:%S ') + f'{run} {base}: {len(nb)} frasi in {time.time()-t0:.0f}s', flush=True)
        if a.once:
            break
        time.sleep(120)


if __name__ == '__main__':
    main()
