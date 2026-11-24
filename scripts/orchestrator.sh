#!/bin/bash
# Ciclo infinito: per ogni run con N-best 5gram pronte (test+competition):
#  1) se manca lo scoring LLM -> accoda in testa un job llm_score (una volta sola)
#  2) se lo scoring c'e' e manca rescore -> esegue rescore_eval e scrive la submission
cd /mnt/20TB/asaccoia/brain2text
LLM=opt67
while true; do
  for d in runs/*/; do r=$(basename $d)
    [ -f $d/nbest_5gram_test.json ] && [ -f $d/nbest_5gram_competition.json ] || continue
    if [ ! -f $d/llm_${LLM}_5gram_test.json ] || [ ! -f $d/llm_${LLM}_5gram_competition.json ]; then
      if ! grep -qF "llm_score.py --once --runs $r " queue/history.txt queue/pending.txt 2>/dev/null; then
        J=".venv/bin/python src/llm_score.py --once --runs $r --tag $LLM --load 8bit "
        flock queue/.lock bash -c "{ echo '$J'; cat queue/pending.txt; } > queue/.tmp && mv queue/.tmp queue/pending.txt"
        echo "$(date '+%F %T') accodato scoring LLM per $r"
      fi
      continue
    fi
    if [ ! -f $d/rescore_5gram_${LLM}.json ]; then
      .venv/bin/python src/rescore_eval.py --run $r --lm 5gram --llm $LLM --write_submission 2>&1 | grep -v Warn
    fi
  done
  sleep 300
done
