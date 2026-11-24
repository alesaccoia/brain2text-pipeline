#!/bin/bash
# Decodifica 5-gram con rescoring G_no_prune (~300 GB RAM), poi scoring OPT e rescore.
# Watchdog: se la RAM disponibile scende sotto 12 GB, uccide la decodifica per proteggere la macchina.
cd /mnt/20TB/asaccoia/brain2text
. scripts/env.sh 2>/dev/null
RUNS=tf_d512_m50_s0,tf_d384_m50_s1,tf_d384_m50_s2,tf_d384_m50_40k_s0,tf_d384_m50_s0,gru_base_s0
.venv-lm/bin/python -u src/decode_ngram.py --lm 5gram --unpruned 1 --once --runs $RUNS &
DP=$!
while kill -0 $DP 2>/dev/null; do
  AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  RSS=$(ps -o rss= -p $DP | awk '{print int($1/1048576)}')
  echo "$(date '+%F %T') RAM disponibile ${AV} GB, decoder ${RSS} GB" >> logs/unpruned_mem.log
  if [ "$AV" -lt 12 ]; then echo "$(date) RAM sotto 12 GB: uccido la decodifica"; kill -9 $DP; exit 1; fi
  sleep 20
done
wait $DP || { echo "decodifica fallita"; exit 1; }
for r in ${RUNS//,/ }; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python src/llm_score.py --once --runs $r --tag opt67 --load 8bit
  .venv/bin/python src/rescore_eval.py --run $r --lm 5gram_unpr --llm opt67 --write_submission 2>&1 | grep -v Warn
done
echo "$(date) FINITO"
