#!/bin/bash
# Worker: prende la prima riga da queue/pending.txt e la esegue sulla GPU $1 (slot $2). Gira per sempre.
# Per aggiungere job: echo "comando" >> queue/pending.txt  (comando eseguito da /mnt/20TB/asaccoia/brain2text)
GPU=$1; SLOT=$2
. /mnt/20TB/asaccoia/brain2text/scripts/env.sh
while true; do
  [ -f queue/STOP ] && { echo "STOP trovato, esco"; exit 0; }
  JOB=$(flock queue/.lock bash -c 'l=$(head -n1 queue/pending.txt 2>/dev/null); [ -n "$l" ] && { sed -i 1d queue/pending.txt; echo "$l"; }')
  if [ -z "$JOB" ]; then sleep 60; continue; fi
  echo "$(date '+%F %T') [gpu$GPU.$SLOT] START $JOB" | tee -a queue/history.txt
  CUDA_VISIBLE_DEVICES=$GPU bash -c "$JOB" > logs/job_gpu${GPU}_${SLOT}_$(date +%s).log 2>&1
  echo "$(date '+%F %T') [gpu$GPU.$SLOT] END rc=$? $JOB" | tee -a queue/history.txt
done
