# brain2text-pipeline

Code for the [Brain-to-Text Benchmark '24](https://eval.ai/web/challenges/challenge-page/2099/overview). The task is to decode sentences that a participant with ALS tried to speak, using intracortical recordings from speech motor cortex (Willett et al., 2023).

The pipeline has five stages:

1. **Neural decoder.** Binned neural features (20 ms) go in, phoneme probabilities come out, trained with CTC. The architecture is a GRU or a Transformer.
2. **WFST decoding.** A 5-gram language model (Kaldi-style T∘L∘G graph) turns each phoneme posteriorgram into a 100-best list of sentences.
3. **LLM rescoring.** OPT-6.7B scores every hypothesis. The final score combines the acoustic, n-gram and LLM scores.
4. **Multi-model fusion.** The top hypotheses of several decoders are passed to an instruction-tuned LLM, which writes the final sentence.
5. **Evaluation and submission.** WER is computed on the test split and a submission file is written for the competition split.

## Results

The table shows word error rate on the public `test` split. Nothing in it has been scored on the competition leaderboard yet.

The official baseline reports its test WER on 600 sentences only (sessions 4–18, see `rnn_step3_baselineRNNInference.ipynb` in `speechBCI`). The full test split has 880 sentences. To compare with the baseline, use the same 600 sentences: the full split is harder.

| System | Test WER, 600-sentence subset |
|---|---|
| Official baseline: GRU + 5-gram + OPT-6.7B | 13.7% |
| Our GRU, same recipe | 15.3% |
| Transformer (d=512, 6 layers) + 5-gram + OPT-6.7B | 11.7% |

Fusing 6 decoders with an LLM (stage 4) reached 9.9% WER on 200 randomly chosen test sentences. That figure is preliminary and is not comparable with the rows above.

Observations from ablations:

- Rescoring with the unpruned 5-gram (`G_no_prune.fst`, ~300 GB RAM) lowers n-gram-only WER by about 1 point. After OPT rescoring the gain shrinks to 0.2–0.4 points.
- Adding area 44 (Broca) channels to area 6v makes the result worse.
- Using all four threshold-crossing features instead of one does not help the Transformer.

## Layout

```
src/
  prep.py           competitionData .mat -> data/processed/b2t24.pkl (block-wise z-scoring, g2p phoneme targets)
  train.py          GRU / Transformer CTC decoder; writes runs/NAME/{config.json,log.txt,best.pt,logits_*.npz}
  decode_ngram.py   WFST n-gram decoding of saved logits -> runs/NAME/nbest_{lm}_{split}.json
  llm_score.py      LLM log-probabilities for each n-best hypothesis
  rescore_eval.py   combines scores, tunes weights on test, writes submissions/*.txt
  llm_merge.py      multi-decoder fusion with an LLM behind an OpenAI-compatible endpoint
scripts/            data download, lm_decoder build, job queue workers, status
paper/              technical report (LaTeX, in Italian) and figures
docs/               operational notes for the machine this was developed on (Italian)
```

## Setup

### Requirements

- Python 3.12 for training and scoring.
- Python 3.10 for the WFST decoder.
- A CUDA GPU. OPT-6.7B in 8-bit needs about 8 GB of VRAM.
- RAM:
  - about 60 GB for the pruned 5-gram;
  - about 300 GB to rescore with the unpruned 5-gram.

### Steps

1. **Install training and scoring dependencies:**

   ```bash
   python -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. **Build the WFST decoder (`lm_decoder`)** from the `LanguageModelDecoder` of [speechBCI](https://github.com/fwillett/speechBCI). Clone that repo into `repos/speechBCI`, then run:

   ```bash
   scripts/build_lm_decoder.sh
   ```

3. **Download the data and language models** from [Dryad (doi:10.5061/dryad.x69p8czpq)](https://doi.org/10.5061/dryad.x69p8czpq) into `data/raw/`, and extract them into `data/extracted/`. Downloading through the Dryad API needs API credentials, which `scripts/dryad_download.sh` reads from `.dryad_credentials`.

4. **Edit the paths.** `ROOT` is hard-coded at the top of each file in `src/` and `scripts/`. Change it to your checkout path.

## Usage

```bash
# 1. preprocess
.venv/bin/python src/prep.py

# 2. train (examples)
.venv/bin/python src/train.py --name gru_base_s0 --seed 0
.venv/bin/python src/train.py --name tf_d512_m50_s0 --arch transformer --optim adamw \
  --lr 1e-3 --lr_end 1e-5 --warmup 1000 --wd 0.05 --dropout 0.2 --steps 20000 \
  --white_noise 0.2 --offset_noise 0.05 --d_model 512 --heads 8 --layers 6 \
  --time_mask_frac 0.5 --seed 0

# 3. n-best decoding (add --unpruned 1 for G_no_prune rescoring)
.venv-lm/bin/python src/decode_ngram.py --lm 5gram --once --runs tf_d512_m50_s0

# 4. LLM rescoring, weight tuning, submission
.venv/bin/python src/llm_score.py --once --runs tf_d512_m50_s0 --tag opt67 --load 8bit
.venv/bin/python src/rescore_eval.py --run tf_d512_m50_s0 --lm 5gram --llm opt67 --write_submission

# 5. multi-model fusion (needs an OpenAI-compatible endpoint, key in PROXY_KEY or .proxy_key)
.venv/bin/python src/llm_merge.py --split test --n 200 --v2 1
.venv/bin/python src/llm_merge.py --split competition --v2 1
```

`train.py --fold k` holds out one fold of training blocks and saves out-of-fold logits. These are meant for training a local fusion model.

## Credits

- **Data, baseline decoder and language models:** Willett, F. R. et al. *A high-performance speech neuroprosthesis.* Nature 620, 1031–1036 (2023). <https://doi.org/10.1038/s41586-023-06377-x>
- **Reference code:**
  - [fwillett/speechBCI](https://github.com/fwillett/speechBCI), for the WFST decoder and the baseline configuration;
  - [cffan/neural_seq_decoder](https://github.com/cffan/neural_seq_decoder).

  Neither is included in this repository.
