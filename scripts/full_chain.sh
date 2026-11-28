#!/bin/bash
# Catena completa per uno o piu' run (o ensemble) gia' con logits: 5-gram + G_no_prune (~300 GB RAM, caricato una volta),
# poi OPT-6.7B e rescore per ciascuno.
# Uso: scripts/full_chain.sh RUN1[,RUN2,...]   (OPT gira sulla GPU con piu memoria libera)
cd /mnt/20TB/asaccoia/brain2text
. scripts/env.sh 2>/dev/null
RUNS=$1; G=${2:-0}
.venv-lm/bin/python -u src/decode_ngram.py --lm 5gram --unpruned 1 --once --runs $RUNS || { echo "decodifica fallita"; exit 1; }
for R in ${RUNS//,/ }; do
  G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
  CUDA_VISIBLE_DEVICES=$G .venv/bin/python src/llm_score.py --once --runs $R --tag opt67 --load 8bit
  .venv/bin/python src/rescore_eval.py --run $R --lm 5gram_unpr --llm opt67 --write_submission 2>&1 | grep -v Warn
done
echo "$(date) FINITO $RUNS"
