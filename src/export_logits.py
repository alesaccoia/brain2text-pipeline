"""Esporta logits test/competition da un checkpoint: python src/export_logits.py SRC_RUN DST_RUN [best|model]"""
import sys, json, os, argparse, torch
sys.path.insert(0, os.path.dirname(__file__))
import train as T
src, dst = sys.argv[1], sys.argv[2]
ck = sys.argv[3] if len(sys.argv) > 3 else 'best'
a = argparse.Namespace(**json.load(open(f'{T.ROOT}/runs/{src}/config.json')))
d = T.load_data(a.data, a.feats)
m = T.Decoder(a, d['train'][0]['x'].shape[1], len(d['sessions'])).cuda()
m.load_state_dict(torch.load(f'{T.ROOT}/runs/{src}/{ck}.pt', map_location='cuda'))
out = f'{T.ROOT}/runs/{dst}'; os.makedirs(out, exist_ok=True)
json.dump(vars(a) | {'exported_from': src, 'ckpt': ck}, open(f'{out}/config.json', 'w'), indent=1)
per, lt = T.run_eval(m, d['test'], 'cuda', want_logits=True)
_, lc = T.run_eval(m, d['competition'], 'cuda', want_logits=True)
T.save_logits(f'{out}/logits_competition.npz', lc, d['competition'])
T.save_logits(f'{out}/logits_test.npz', lt, d['test'])
open(f'{out}/log.txt', 'w').write(f'export da {src}/{ck}.pt: PER test {per:.4f}\n')
print('PER test', per)
