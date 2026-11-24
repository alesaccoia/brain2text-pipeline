#!/bin/bash
# Riepilogo dello stato della pipeline Brain-to-Text
cd /mnt/20TB/asaccoia/brain2text
echo "=== $(date '+%F %T')  screen:"; screen -ls | grep b2t | awk '{print "  "$1}'
echo "=== GPU"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "=== RAM"; free -g | awk 'NR==2{print "  usata "$3" GB, disponibile "$7" GB"}'
echo "=== coda: $(wc -l < queue/pending.txt) job in attesa"
echo "=== run (ultima riga log | risultati)"
for d in runs/*/; do r=$(basename $d)
  printf "  %-28s %s" $r "$(tail -n1 $d/log.txt 2>/dev/null | cut -c1-70)"
  for w in $d/wer_*_test.json $d/rescore_*.json; do [ -f $w ] && printf " | %s %s" $(basename $w .json) "$(cat $w)"; done; echo; done
echo "=== decodifica n-gram"; tail -n3 logs/decode_5gram.log 2>/dev/null
echo "=== LLM scoring"; tail -n3 logs/llm_score.log 2>/dev/null
