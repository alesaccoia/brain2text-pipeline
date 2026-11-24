#!/bin/bash
# Tabella PER test (greedy) dei run: ultimo valore, migliore, passo. Uso: scripts/per_table.sh [prefisso]
cd /mnt/20TB/asaccoia/brain2text
for d in runs/${1:-}*/; do r=$(basename $d); [ -f $d/log.txt ] || continue
  last=$(grep -o "step [0-9]* .*test_PER [0-9.]*" $d/log.txt | tail -1 | awk '{print $2, $NF}')
  res=$( [ -f $d/result.json ] && cat $d/result.json || echo "in corso")
  printf "%-28s %-16s %s\n" "$r" "$last" "$res"
done
