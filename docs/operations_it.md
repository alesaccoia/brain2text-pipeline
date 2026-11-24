# Brain-to-Text '24 (EvalAI 2099)

Obiettivo: WER sotto il 2,99% (primo in classifica al 22/11/2026). La gara chiude il 13/01/2027. Si possono inviare al massimo 5 sottomissioni al giorno.

## Stato in un colpo d'occhio

```bash
/mnt/20TB/asaccoia/brain2text/scripts/status.sh
```

## Pipeline

1. `src/prep.py` converte `data/extracted/competitionData` in `data/processed/b2t24.pkl`. Usa tutte le feature, z-scored per blocco, e i target fonemici ottenuti con g2p_en.
2. `src/train.py` addestra il decoder neurale → fonemi (CTC) con un'architettura GRU o Transformer. Scrive in `runs/NOME/` i logits di test e competition.
3. `src/decode_ngram.py` decodifica i logits con il WFST 5-gram e produce le N-best (100). Gira come servizio in `.venv-lm`, che contiene `lm_decoder` compilato.
4. `src/llm_score.py` calcola il log-prob di un LLM (OPT-6.7B) per ogni ipotesi.
5. `src/rescore_eval.py` combina acustica, n-gram e LLM, tara i pesi sul test e scrive `submissions/*.txt`.
6. Ultimo passo, da fare: ensemble di molti decoder, fusi da un LLM addestrato con QLoRA.

## Sessioni screen

| screen | cosa |
|---|---|
| `b2t-v2-gXY` | worker di addestramento (GPU X, slot Y). Prendono i job da `queue/pending.txt` |
| `b2t-decode` | servizio di decodifica 5-gram (circa 50 GB di RAM) |
| `b2t-orch` | orchestratore: accoda lo scoring LLM (OPT-6.7B 8bit su GPU) e scrive le submission |
| `b2t-build` | compilazione di `lm_decoder` (termina da sola) |

- **Aggiungere un job:** `echo ".venv/bin/python src/train.py --name X ..." >> queue/pending.txt`
- **Fermare i worker dopo i job in corso:** `touch queue/STOP2`
- **Storico dei job:** `queue/history.txt`. I log sono in `logs/`.

## Note

- Il driver NVIDIA è il 535 (CUDA 12.2), quindi torch va tenuto alla versione 2.7.1+cu126. Usa `pip install -c constraints.txt`.
- **5-gram senza `G_no_prune`:** il file da 80 GB richiederebbe circa 300 GB di RAM, con il rischio che il sistema chiuda per memoria i processi krktrader.
- **Credenziali Dryad:** sono in `.dryad_credentials`. Non vanno committate.
