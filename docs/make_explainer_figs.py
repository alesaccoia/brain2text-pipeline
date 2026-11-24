"""Figures for docs/EXPLAINER.md, generated from the real data and runs (PNG in docs/img)."""
import glob, json, pickle, re, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

ROOT = '/mnt/20TB/asaccoia/brain2text'
OUT = f'{ROOT}/docs/img'
sys.path.insert(0, f'{ROOT}/src')
from rescore_eval import wer, pick, clean  # noqa: E402

S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']  # categorical, fixed order
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'text.color': INK, 'savefig.bbox': 'tight', 'savefig.dpi': 160, 'figure.facecolor': 'white'})

d = pickle.load(open(f'{ROOT}/data/processed/b2t24.pkl', 'rb'))
PH = d['phone_def']
LAB = ['blank'] + PH
EX = 5  # test sentence used in several figures


def save(fig, name):
    fig.savefig(f'{OUT}/{name}.png'); plt.close(fig)


# 1. neural activity for one sentence ------------------------------------------------
t = d['test'][EX]
x = gaussian_filter1d(t['x'].astype(np.float32), 2.0, axis=0)
fig, ax = plt.subplots(2, 1, figsize=(7.5, 3.8), sharex=True)
T = x.shape[0]
for k, (lab, c0) in enumerate([('Threshold crossings (tx1), area 6v, 128 electrodes', 0),
                               ('Spike-band power, area 6v, 128 electrodes', 1024)]):
    im = ax[k].imshow(x[:, c0:c0 + 128].T, aspect='auto', cmap='RdBu_r', vmin=-1.2, vmax=1.2,
                      extent=[0, T * 0.02, 128, 0], interpolation='nearest')
    ax[k].set_ylabel('electrode'); ax[k].set_title(lab, fontsize=9, loc='left')
ax[1].set_xlabel('time from go cue (s)')
fig.colorbar(im, ax=ax, shrink=0.8, label='z-score (smoothed)')
fig.suptitle(f'Input X for the sentence “{t["text"]}”', fontsize=10, x=0.42, y=1.0)
save(fig, 'neural_activity')

# 2. dataset by day -----------------------------------------------------------------------
days = d['sessions']
cnt = {p: np.bincount([u['day'] for u in d[p]], minlength=len(days)) for p in ['train', 'test', 'competition']}
fig, ax = plt.subplots(figsize=(7.5, 2.6))
bottom = np.zeros(len(days))
for (p, lab), c in zip([('train', f'train ({len(d["train"])})'), ('test', f'test ({len(d["test"])})'),
                        ('competition', f'competition, labels hidden ({len(d["competition"])})')], S):
    ax.bar(range(len(days)), cnt[p], bottom=bottom, color=c, label=lab, width=0.8, edgecolor='white', linewidth=1)
    bottom += cnt[p]
ax.set_xticks(range(len(days))); ax.set_xticklabels([s[4:] for s in days], rotation=70, fontsize=7)
ax.set_ylabel('sentences'); ax.set_xlabel('recording session (2022)')
ax.legend(frameon=False, ncol=3, fontsize=8, loc='upper left', bbox_to_anchor=(0, 1.18))
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
save(fig, 'dataset_by_day')

# 3. CTC posteriors of the best model for the same sentence ----------------------------
z = np.load(f'{ROOT}/runs/tf_d512_m50_s0/logits_test.npz', allow_pickle=True)
P = np.exp(z['logits'][EX].astype(np.float32))
best = P.argmax(1)
used = sorted(set(best.tolist()) | set([PH.index(p) + 1 for p in t['phonemes'] if p in PH]
                                         if isinstance(t['phonemes'][0], str) else set()))
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.imshow(P[:, used].T, aspect='auto', cmap='Greys', extent=[0, P.shape[0] * 0.08, len(used), 0],
          interpolation='nearest', vmin=0, vmax=1)
ax.set_yticks(np.arange(len(used)) + 0.5); ax.set_yticklabels([LAB[c] for c in used], fontsize=7)
ax.set_xlabel('time (s), one output frame every 80 ms'); ax.set_ylabel('class')
prev = -1
for ti, c in enumerate(best):
    if c != prev and c != 0:
        ax.text(ti * 0.08 + 0.1, used.index(c) + 0.5, LAB[c], color=S[1], fontsize=7, ha='left', va='center',
                weight='bold')
    prev = c
ax.set_title('Per-frame class probabilities (dark = high). Orange: phonemes emitted by greedy decoding.\n'
             'Most frames are "blank"; each phoneme is a short spike.', fontsize=8.5, loc='left')
save(fig, 'ctc_posteriors')
greedy = []
prev = -1
for c in best:
    if c != prev and c != 0:
        greedy.append('|' if LAB[c] == 'SIL' else LAB[c])
    prev = c

# 4. training curves -----------------------------------------------------------------------
runs = [('gru_base_s0', 'GRU 5x1024 (baseline recipe)'), ('gru_6v44_s0', 'GRU, areas 6v + 44'),
        ('tf_d384_m50_s0', 'Transformer d=384'), ('tf_d512_m50_s0', 'Transformer d=512')]
fig, ax = plt.subplots(figsize=(7.5, 3.0))
for (run, lab), c in zip(runs, S):
    s, p = [], []
    for line in open(f'{ROOT}/runs/{run}/log.txt'):
        m = re.search(r'step (\d+) .*test_PER ([\d.]+)', line)
        if m:
            s.append(int(m.group(1))); p.append(float(m.group(2)) * 100)
    ax.plot(s, p, color=c, lw=2, label=lab)
    dy = {'tf_d384_m50_s0': 5, 'tf_d512_m50_s0': -5}.get(run, 0)
    ax.annotate(f'{p[-1]:.1f}%', (s[-1], p[-1]), xytext=(4, dy), textcoords='offset points', fontsize=7.5,
                color=INK2, va='center')
ax.set_ylim(15, 50); ax.set_xlabel('training step'); ax.set_ylabel('phoneme error rate on test (%)')
ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8)
save(fig, 'training_curves')

# 5. WER by stage on the baseline's 600-sentence subset ----------------------------------
meta = json.loads(str(z['meta']))
refs = [m['text'] for m in meta]
ud = sorted(set(m['day'] for m in meta))
sel = [i for i, m in enumerate(meta) if 4 <= ud.index(m['day']) <= 18]


def stage_wers(run):
    out = []
    for lm in ['5gram', '5gram_unpr']:
        nb = json.load(open(f'{ROOT}/runs/{run}/nbest_{lm}_test.json'))
        out.append(wer([refs[i] for i in sel], [clean(nb[i][0][0]) if nb[i] else '' for i in sel]))
    nb = json.load(open(f'{ROOT}/runs/{run}/nbest_5gram_unpr_test.json'))
    ll = json.load(open(f'{ROOT}/runs/{run}/llm_opt67_5gram_unpr_test.json'))
    r = json.load(open(f'{ROOT}/runs/{run}/rescore_5gram_unpr_opt67.json'))
    h = pick(nb, ll, r['ac_scale'], r['alpha'])
    out.append(wer([refs[i] for i in sel], [clean(h[i]) for i in sel]))
    return [100 * v for v in out]


stages = ['5-gram (pruned)', '+ unpruned 5-gram\nrescoring', '+ OPT-6.7B\nrescoring']
W = {'GRU (baseline recipe)': stage_wers('gru_base_s0'), 'Transformer d=512': stage_wers('tf_d512_m50_s0')}
fig, ax = plt.subplots(figsize=(7.5, 3.0))
xs = np.arange(len(stages)); w = 0.36
for k, ((lab, v), c) in enumerate(zip(W.items(), S)):
    b = ax.bar(xs + (k - 0.5) * w, v, w * 0.94, color=c, label=lab)
    for xi, vi in zip(xs + (k - 0.5) * w, v):
        ax.text(xi, vi + 0.3, f'{vi:.1f}', ha='center', fontsize=7.5, color=INK2)
ax.axhline(13.7, color=INK2, lw=1, ls='--', label='official baseline, final (13.7%)')
ax.set_xticks(xs); ax.set_xticklabels(stages); ax.set_ylabel('word error rate (%)')
ax.set_title('Test WER on the 600 sentences the baseline reports (sessions 4-18)', fontsize=9, loc='left')
ax.set_ylim(0, 23); ax.set_xlim(-0.6, 2.6); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc='upper right')
save(fig, 'wer_by_stage')

# 6. a worked example through the whole chain ---------------------------------------------
mg = json.load(open(f'{ROOT}/runs/merge_astra_test.json'))
fused = dict(zip(mg['ev'], mg['hyp']))
nb = json.load(open(f'{ROOT}/runs/tf_d512_m50_s0/nbest_5gram_unpr_test.json'))
ll = json.load(open(f'{ROOT}/runs/tf_d512_m50_s0/llm_opt67_5gram_unpr_test.json'))
r = json.load(open(f'{ROOT}/runs/tf_d512_m50_s0/rescore_5gram_unpr_opt67.json'))
opt = pick(nb, ll, r['ac_scale'], r['alpha'])
examples = []
for i in mg['ev']:
    ref = clean(refs[i]); ng = clean(nb[i][0][0]); op = clean(opt[i]); fu = fused[i]
    if ng != op and op != fu and fu == ref and len(ref.split()) >= 4:
        lp = z['logits'][i].astype(np.float32).argmax(1)
        g, prev = [], -1
        for c in lp:
            if c != prev and c != 0:
                g.append('|' if LAB[c] == 'SIL' else LAB[c])
            prev = c
        examples.append({'ref': ref, 'greedy': ' '.join(g), 'ngram_top3': [clean(h[0]) for h in nb[i][:3]],
                         'opt': op, 'fused': fu})
json.dump({'ctc_example': {'ref': t['text'], 'greedy': ' '.join(greedy)}, 'chain_examples': examples[:5]},
          open(f'{OUT}/examples.json', 'w'), indent=1)
print('done;', len(examples), 'chain examples')
