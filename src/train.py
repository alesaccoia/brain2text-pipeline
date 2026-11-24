"""Addestra un decoder neurale -> fonemi (CTC) sul dataset Brain-to-Text '24.

Uso: python src/train.py --name NOME [--arch gru|transformer] [--seed 0] ...
Salva in runs/NOME: config.json, log.txt, model.pt (finale), best.pt, e logits_{test,competition}.npz
(e logits_train.npz se --save_train_logits).
"""
import argparse, json, math, os, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = '/mnt/20TB/asaccoia/brain2text'

FEATSETS = {
    # colonne in [tx1 | tx2 | tx3 | tx4 | spikePow], 256 canali ciascuno (0:128 = area 6v)
    '6v': list(range(0, 128)) + list(range(1024, 1152)),
    '6v44': list(range(0, 256)) + list(range(1024, 1280)),
    '6v_alltx': [g * 256 + c for g in range(5) for c in range(128)],
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--data', default=f'{ROOT}/data/processed/b2t24.pkl')
    p.add_argument('--arch', default='gru', choices=['gru', 'transformer', 'conformer'])
    p.add_argument('--feats', default='6v', choices=list(FEATSETS))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--lr_end', type=float, default=0.002)
    p.add_argument('--warmup', type=int, default=0)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--optim', default='adam', choices=['adam', 'adamw'])
    p.add_argument('--hidden', type=int, default=1024)
    p.add_argument('--layers', type=int, default=5)
    p.add_argument('--dropout', type=float, default=0.4)
    p.add_argument('--bidir', type=int, default=1)
    p.add_argument('--layernorm', type=int, default=1)
    p.add_argument('--kernel', type=int, default=32)
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--smooth', type=float, default=2.0)
    p.add_argument('--white_noise', type=float, default=0.8)
    p.add_argument('--offset_noise', type=float, default=0.2)
    p.add_argument('--time_mask_frac', type=float, default=0.0, help='frazione di tempo mascherata (somma delle maschere)')
    p.add_argument('--time_mask_len', type=int, default=20, help='lunghezza max di ogni maschera in bin da 20 ms')
    p.add_argument('--speckle', type=float, default=0.0, help='prob. di azzerare singoli elementi input')
    p.add_argument('--d_model', type=int, default=384)
    p.add_argument('--heads', type=int, default=6)
    p.add_argument('--train_on_test', type=int, default=0, help='1 = aggiunge la partizione test al train (modello finale)')
    p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--save_train_logits', type=int, default=0)
    p.add_argument('--fold', type=int, default=-1, help='>=0: esclude dal train i blocchi del fold e salva logits_oof.npz')
    p.add_argument('--nfolds', type=int, default=5)
    # --- v2: miglioramenti del decoder neurale
    p.add_argument('--ema', type=float, default=0.0, help='decadimento EMA dei pesi (0 = off); eval e salvataggi usano i pesi EMA')
    p.add_argument('--interctc', type=float, default=0.0, help='peso della CTC intermedia (strato a meta)')
    p.add_argument('--time_warp', type=float, default=0.0, help='fattore max di velocita (es. 0.15 = 0.85..1.15)')
    p.add_argument('--chan_drop', type=float, default=0.0, help='prob. di azzerare interi canali per frase')
    p.add_argument('--conv_kernel', type=int, default=15, help='kernel della convoluzione depthwise (conformer)')
    p.add_argument('--layerdrop', type=float, default=0.0, help='stochastic depth: prob. max di saltare un blocco')
    return p.parse_args()


# ----------------------------------------------------------------------------- dati
def load_data(path, featset):
    with open(path, 'rb') as fh:
        d = pickle.load(fh)
    cols = FEATSETS[featset]
    for part in ['train', 'test', 'competition']:
        for t in d[part]:
            t['x'] = torch.from_numpy(np.ascontiguousarray(t['x'][:, cols]))
            if 'phonemes' in t:
                t['y'] = torch.from_numpy(t['phonemes'])
    return d


def collate(trials, device):
    lens = torch.tensor([t['x'].shape[0] for t in trials])
    X = torch.zeros(len(trials), int(lens.max()), trials[0]['x'].shape[1], dtype=torch.float16)
    for i, t in enumerate(trials):
        X[i, :lens[i]] = t['x']
    days = torch.tensor([t['day'] for t in trials])
    out = dict(X=X.to(device).float(), lens=lens.to(device), days=days.to(device))
    if 'y' in trials[0]:
        ylens = torch.tensor([len(t['y']) for t in trials])
        Y = torch.zeros(len(trials), int(ylens.max()), dtype=torch.long)
        for i, t in enumerate(trials):
            Y[i, :ylens[i]] = t['y']
        out.update(Y=Y.to(device), ylens=ylens.to(device))
    return out


# ----------------------------------------------------------------------------- modello
def gaussian_kernel(width, size=20):
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    k = torch.exp(-0.5 * (x / width) ** 2)
    return k / k.sum()


class ConformerBlock(nn.Module):
    def __init__(self, d, heads, kernel, drop):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Dropout(drop), nn.Linear(4 * d, d), nn.Dropout(drop))
        self.ln_att = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.drop_att = nn.Dropout(drop)
        self.ln_conv = nn.LayerNorm(d)
        self.pw1 = nn.Conv1d(d, 2 * d, 1)
        self.dw = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.bn = nn.BatchNorm1d(d)
        self.pw2 = nn.Conv1d(d, d, 1)
        self.drop_conv = nn.Dropout(drop)
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.SiLU(), nn.Dropout(drop), nn.Linear(4 * d, d), nn.Dropout(drop))
        self.ln_out = nn.LayerNorm(d)

    def forward(self, x, mask):
        x = x + 0.5 * self.ff1(x)
        h = self.ln_att(x)
        h, _ = self.att(h, h, h, key_padding_mask=mask, need_weights=False)
        x = x + self.drop_att(h)
        h = self.ln_conv(x).masked_fill(mask[..., None], 0).transpose(1, 2)
        h = F.glu(self.pw1(h), dim=1)
        h = self.pw2(F.silu(self.bn(self.dw(h))))
        x = x + self.drop_conv(h.transpose(1, 2))
        x = x + 0.5 * self.ff2(x)
        return self.ln_out(x)


class Decoder(nn.Module):
    def __init__(self, a, n_in, n_days, n_out=41):
        super().__init__()
        self.a = a
        self.kernel, self.stride = a.kernel, a.stride
        if a.smooth > 0:
            self.register_buffer('gk', gaussian_kernel(a.smooth).view(1, 1, -1))
        self.day_w = nn.Parameter(torch.eye(n_in).repeat(n_days, 1, 1))
        self.day_b = nn.Parameter(torch.zeros(n_days, 1, n_in))
        in_dim = n_in * a.kernel
        if a.arch == 'gru':
            self.in_norm = nn.LayerNorm(in_dim) if a.layernorm else nn.Identity()
            self.rnn = nn.GRU(in_dim, a.hidden, a.layers, batch_first=True, dropout=a.dropout, bidirectional=bool(a.bidir))
            for n_, p_ in self.rnn.named_parameters():
                if 'weight_hh' in n_:
                    nn.init.orthogonal_(p_)
                elif 'weight_ih' in n_:
                    nn.init.xavier_uniform_(p_)
            out_dim = a.hidden * (2 if a.bidir else 1)
            self.out_norm = nn.LayerNorm(out_dim) if a.layernorm else nn.Identity()
        else:
            self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, a.d_model), nn.Dropout(a.dropout))
            self.pos = nn.Parameter(torch.zeros(1, 1024, a.d_model))
            nn.init.normal_(self.pos, std=0.02)
            if a.arch == 'transformer':
                self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(a.d_model, a.heads, 4 * a.d_model, a.dropout,
                                             activation='gelu', batch_first=True, norm_first=True) for _ in range(a.layers)])
            else:
                self.blocks = nn.ModuleList([ConformerBlock(a.d_model, a.heads, a.conv_kernel, a.dropout)
                                             for _ in range(a.layers)])
            out_dim = a.d_model
            self.out_norm = nn.LayerNorm(out_dim)
            if a.interctc > 0:
                self.inter_norm = nn.LayerNorm(out_dim)
                self.inter_fc = nn.Linear(out_dim, n_out)
        self.drop = nn.Dropout(a.dropout)
        self.fc = nn.Linear(out_dim, n_out)

    def out_lens(self, lens):
        return torch.clamp((lens - self.kernel) // self.stride + 1, min=1)

    def forward(self, X, days, lens):
        if self.a.smooth > 0:
            B, T, C = X.shape
            xs = X.permute(0, 2, 1).reshape(B * C, 1, T)
            xs = F.conv1d(xs, self.gk, padding='same')
            X = xs.view(B, C, T).permute(0, 2, 1)
        X = torch.einsum('btd,bdk->btk', X, self.day_w[days]) + self.day_b[days]
        X = F.softsign(X)
        # finestre (kernel, stride) concatenate
        X = X.unfold(1, self.kernel, self.stride)  # B, T', C, K
        X = X.permute(0, 1, 3, 2).reshape(X.shape[0], X.shape[1], -1)
        olens = self.out_lens(lens)
        if self.a.arch == 'gru':
            X = self.in_norm(X)
            X, _ = self.rnn(X)
        else:
            X = self.proj(X)
            X = X + self.pos[:, :X.shape[1]]
            mask = torch.arange(X.shape[1], device=X.device)[None] >= olens[:, None]
            nl = len(self.blocks)
            for i, blk in enumerate(self.blocks):
                pdrop = self.a.layerdrop * (i + 1) / nl
                if not (self.training and pdrop > 0 and torch.rand(1).item() < pdrop):
                    X = blk(X, src_key_padding_mask=mask) if self.a.arch == 'transformer' else blk(X, mask)
                if self.a.interctc > 0 and i == nl // 2 - 1:
                    self.inter_out = self.inter_fc(self.inter_norm(X))
        X = self.drop(self.out_norm(X))
        return self.fc(X), olens


# ----------------------------------------------------------------------------- augment
def time_warp(X, lens, a, kernel):
    """Ricampiona ogni frase con un fattore di velocita casuale (interpolazione lineare)."""
    out, nl = [], []
    for i in range(X.shape[0]):
        L = int(lens[i])
        f = 1.0 + (2 * np.random.rand() - 1) * a.time_warp
        L2 = max(kernel + 1, int(round(L * f)))
        x = F.interpolate(X[i:i + 1, :L].transpose(1, 2), size=L2, mode='linear', align_corners=False)
        out.append(x[0].T); nl.append(L2)
    T2 = max(nl)
    Y = X.new_zeros(X.shape[0], T2, X.shape[2])
    for i, x in enumerate(out):
        Y[i, :x.shape[0]] = x
    return Y, torch.tensor(nl, device=lens.device)


def augment(X, lens, a):
    if a.white_noise > 0:
        X = X + torch.randn_like(X) * a.white_noise
    if a.offset_noise > 0:
        X = X + torch.randn(X.shape[0], 1, X.shape[2], device=X.device) * a.offset_noise
    if a.speckle > 0:
        X = X * (torch.rand_like(X) > a.speckle)
    if a.chan_drop > 0:
        X = X * (torch.rand(X.shape[0], 1, X.shape[2], device=X.device) > a.chan_drop)
    if a.time_mask_frac > 0:
        for i in range(X.shape[0]):
            L = int(lens[i])
            budget = int(L * a.time_mask_frac)
            while budget > 0:
                m = int(np.random.randint(1, a.time_mask_len + 1))
                s = int(np.random.randint(0, max(1, L - m)))
                X[i, s:s + m] = 0
                budget -= m
    return X


# ----------------------------------------------------------------------------- eval
def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


@torch.no_grad()
def run_eval(model, trials, device, bs=64, want_logits=False):
    model.eval()
    ed = n = 0
    logits = []
    for i in range(0, len(trials), bs):
        b = collate(trials[i:i + bs], device)
        with torch.autocast('cuda', dtype=torch.float16):
            out, olens = model(b['X'], b['days'], b['lens'])
        lp = out.float().log_softmax(-1)
        for j in range(lp.shape[0]):
            L = int(olens[j])
            if want_logits:
                logits.append(lp[j, :L].cpu().numpy().astype(np.float16))
            if 'Y' in b:
                pred = torch.unique_consecutive(lp[j, :L].argmax(-1)).tolist()
                pred = [p for p in pred if p != 0]
                true = b['Y'][j, :int(b['ylens'][j])].tolist()
                ed += edit_distance(true, pred)
                n += len(true)
    model.train()
    return (ed / n if n else float('nan')), logits


def save_logits(path, logits, trials):
    meta = [{k: t[k] for k in ['day', 'block', 'order', 'partition'] if k in t} | ({'text': t['text']} if 'text' in t else {})
            for t in trials]
    np.savez_compressed(path, logits=np.array(logits, dtype=object), meta=np.array(json.dumps(meta)))


# ----------------------------------------------------------------------------- main
def main():
    a = get_args()
    out_dir = f'{ROOT}/runs/{a.name}'
    os.makedirs(out_dir, exist_ok=True)
    json.dump(vars(a), open(f'{out_dir}/config.json', 'w'), indent=1)
    logf = open(f'{out_dir}/log.txt', 'a')

    def log(*s):
        msg = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in s)
        print(msg, flush=True)
        logf.write(msg + '\n'); logf.flush()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = 'cuda'
    d = load_data(a.data, a.feats)
    oof = []
    if a.fold >= 0:
        # fold per blocco: dentro ogni giorno i blocchi di train sono assegnati a rotazione ai fold
        fold_of = {}
        for day in sorted({t['day'] for t in d['train']}):
            blocks = sorted({t['block'] for t in d['train'] if t['day'] == day})
            for i, b in enumerate(blocks):
                fold_of[(day, b)] = (i + day) % a.nfolds
        oof = [t for t in d['train'] if fold_of[(t['day'], t['block'])] == a.fold]
        d['train'] = [t for t in d['train'] if fold_of[(t['day'], t['block'])] != a.fold]
    train = d['train'] + (d['test'] if a.train_on_test else [])
    test, comp = d['test'], d['competition']
    n_in = train[0]['x'].shape[1]
    model = Decoder(a, n_in, len(d['sessions'])).to(device)
    log(f'parametri: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, train={len(train)} test={len(test)} comp={len(comp)}')

    if a.optim == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.999), eps=0.1, weight_decay=a.wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=a.wd)

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        frac = (s - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr_end + (a.lr - a.lr_end) * (1 - frac)  # decadimento lineare

    scaler = torch.amp.GradScaler(enabled=bool(a.amp))
    ema = None
    if a.ema > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema), use_buffers=True)

    def emodel():
        return ema.module if ema is not None else model
    ctc = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    best = 1e9
    t0 = time.time()
    for step in range(a.steps):
        for g in opt.param_groups:
            g['lr'] = lr_at(step)
        idx = np.random.randint(0, len(train), a.batch)
        b = collate([train[i] for i in idx], device)
        X, lens = b['X'], b['lens']
        if a.time_warp > 0:
            X, lens = time_warp(X, lens, a, a.kernel)
        X = augment(X, lens, a)
        with torch.autocast('cuda', dtype=torch.float16, enabled=bool(a.amp)):
            out, olens = model(X, b['days'], lens)
        loss = ctc(out.float().log_softmax(-1).transpose(0, 1), b['Y'], olens, b['ylens'])
        if a.interctc > 0 and a.arch != 'gru':
            li = ctc(model.inter_out.float().log_softmax(-1).transpose(0, 1), b['Y'], olens, b['ylens'])
            loss = (1 - a.interctc) * loss + a.interctc * li
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(opt); scaler.update()
        if ema is not None:
            ema.update_parameters(model)
        if step % a.eval_every == 0 or step == a.steps - 1:
            per, _ = run_eval(emodel(), test, device)
            log(f'step {step} loss {loss.item():.3f} test_PER {per:.4f} lr {lr_at(step):.5f} {(time.time()-t0)/(step+1):.3f}s/step')
            if per < best and not a.train_on_test:
                best = per
                torch.save(emodel().state_dict(), f'{out_dir}/best.pt')
    model = emodel()
    torch.save(model.state_dict(), f'{out_dir}/model.pt')
    per, lt = run_eval(model, test, device, want_logits=True)
    _, lc = run_eval(model, comp, device, want_logits=True)
    save_logits(f'{out_dir}/logits_test.npz', lt, test)
    save_logits(f'{out_dir}/logits_competition.npz', lc, comp)
    if oof:
        per_oof, lo = run_eval(model, oof, device, want_logits=True)
        save_logits(f'{out_dir}/logits_oof.npz', lo, oof)
        log(f'PER oof {per_oof:.4f} su {len(oof)} frasi')
    if a.save_train_logits:
        _, ltr = run_eval(model, d['train'], device, want_logits=True)
        save_logits(f'{out_dir}/logits_train.npz', ltr, d['train'])
    json.dump({'final_test_PER': per, 'best_test_PER': best}, open(f'{out_dir}/result.json', 'w'))
    log(f'FINE: PER test finale {per:.4f} (migliore {best:.4f})')


if __name__ == '__main__':
    main()
