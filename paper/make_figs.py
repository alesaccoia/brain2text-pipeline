"""Genera le figure del paper dai dati e dai run reali (PDF vettoriali in paper/figs)."""
import glob, json, pickle, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/mnt/20TB/asaccoia/brain2text'
FIG = f'{ROOT}/paper/figs'
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight', 'figure.dpi': 150})
def save(fig, name):
    fig.savefig(f'{FIG}/{name}.pdf'); fig.savefig(f'{FIG}/{name}.png', dpi=110)


C = ['#2a6f97', '#e07a5f', '#3d9970', '#8e6c8a', '#c9a227']

d = pickle.load(open(f'{ROOT}/data/processed/b2t24.pkl', 'rb'))
PH = d['phone_def']

# --- 1. esempio di attivita' neurale -------------------------------------------------
t = d['test'][5]
x = t['x'].astype(np.float32)
from scipy.ndimage import gaussian_filter1d
x = gaussian_filter1d(x, 2.0, axis=0)  # stesso smoothing gaussiano (sigma=40 ms) usato dal modello
fig, ax = plt.subplots(2, 1, figsize=(7, 3.6), sharex=True)
T = x.shape[0]
for k, (lab, c0) in enumerate([('Soglia (tx1), area 6v', 0), ('Spike-band power, area 6v', 1024)]):
    im = ax[k].imshow(x[:, c0:c0 + 128].T, aspect='auto', cmap='RdBu_r', vmin=-1.2, vmax=1.2,
                      extent=[0, T * 0.02, 128, 0], interpolation='nearest')
    ax[k].set_ylabel('canale'); ax[k].set_title(lab, fontsize=9, loc='left')
ax[1].set_xlabel('tempo (s)')
fig.colorbar(im, ax=ax, shrink=0.8, label='z-score')
fig.suptitle(f'Frase: “{t["text"]}”', fontsize=9, y=1.01)
save(fig, 'neural_example'); plt.close(fig)

# --- 2. statistiche del dataset -------------------------------------------------------
days = d['sessions']
cnt = {p: np.bincount([u['day'] for u in d[p]], minlength=len(days)) for p in ['train', 'test', 'competition']}
fig, ax = plt.subplots(figsize=(7, 2.4))
bottom = np.zeros(len(days))
for (p, lab), c in zip([('train', 'train'), ('test', 'test'), ('competition', 'competition hold-out')], C):
    ax.bar(range(len(days)), cnt[p], bottom=bottom, color=c, label=lab, width=0.8)
    bottom += cnt[p]
ax.set_xticks(range(len(days))); ax.set_xticklabels([s[4:] for s in days], rotation=70, fontsize=7)
ax.set_ylabel('frasi'); ax.legend(frameon=False, ncol=3, fontsize=8)
save(fig, 'dataset_days'); plt.close(fig)

lens = np.array([u['x'].shape[0] * 0.02 for u in d['train']])
nw = np.array([len(u['text'].split()) for u in d['train']])
fig, ax = plt.subplots(1, 2, figsize=(7, 2.2))
ax[0].hist(lens, bins=50, color=C[0]); ax[0].set_xlabel('durata della frase (s)'); ax[0].set_ylabel('frasi')
ax[1].hist(nw, bins=range(1, 30), color=C[1]); ax[1].set_xlabel('parole per frase')
save(fig, 'dataset_hist'); plt.close(fig)

# --- 3. curve di addestramento -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 2.8))
for i, f in enumerate(sorted(glob.glob(f'{ROOT}/runs/*/log.txt'))):
    run = f.split('/')[-2]
    if run.startswith('check'):
        continue
    s, p = [], []
    for line in open(f):
        m = re.search(r'step (\d+) .*test_PER ([\d.]+)', line)
        if m:
            s.append(int(m.group(1))); p.append(float(m.group(2)))
    if s:
        ax.plot(s, p, label=run, color=C[i % len(C)], lw=1.4)
ax.set_ylim(0.15, 0.6); ax.set_xlabel('passo di ottimizzazione'); ax.set_ylabel('PER sul test (greedy)')
ax.legend(frameon=False, fontsize=7); ax.grid(alpha=0.3)
save(fig, 'training_curves'); plt.close(fig)

# --- 4. posteriori CTC per una frase --------------------------------------------------
z = np.load(f'{ROOT}/runs/check_tf384_best/logits_test.npz', allow_pickle=True)
lp = z['logits'][5].astype(np.float32)
P = np.exp(lp)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.imshow(P.T, aspect='auto', cmap='Greys', extent=[0, P.shape[0] * 0.08, 41, 0], interpolation='nearest')
labels = ['blank'] + PH
ax.set_yticks(np.arange(41) + 0.5); ax.set_yticklabels(labels, fontsize=4.5)
ax.set_xlabel('tempo (s), un frame ogni 80 ms'); ax.set_ylabel('classe')
best = P.argmax(1)
prev = -1
for ti, c in enumerate(best):
    if c != prev and c != 0:
        ax.text(ti * 0.08 + 0.35, c + 0.5, labels[c], color=C[1], fontsize=6, ha='left', va='center', weight='bold')
    prev = c
ax.set_title('Posteriori CTC del decoder Transformer (in rosso i fonemi emessi dalla decodifica greedy)', fontsize=8, loc='left')
save(fig, 'ctc_posteriors'); plt.close(fig)

# --- 5. esempi di decodifica ---------------------------------------------------------
meta = json.loads(str(z['meta']))
nbp = f'{ROOT}/runs/check_tf384_best/nbest_5gram_test.json'
ex = []
nb = json.load(open(nbp)) if os.path.exists(nbp) else None
for i in [5, 40, 123, 300, 512, 777]:
    lpi = z['logits'][i].astype(np.float32)
    g = [int(c) for j, c in enumerate(lpi.argmax(1)) if c != 0 and (j == 0 or c != lpi.argmax(1)[j - 1])]
    phon = ' '.join('|' if labels[c] == 'SIL' else labels[c] for c in g)
    ex.append({'ref': meta[i]['text'], 'greedy': phon, 'ngram': nb[i][0][0] if nb and nb[i] else None,
               'nbest': [h[0] for h in nb[i][:4]] if nb else None})
json.dump(ex, open(f'{FIG}/examples.json', 'w'), indent=1)
print('figure scritte')
