#!/bin/bash
# Worker v2 per GPU $1 slot $2. Aspetta che il vecchio worker dello stesso slot sia uscito.
# Job da queue/pending.txt (una riga = un comando, eseguito da /mnt/20TB/asaccoia/brain2text).
# In caso di CUDA OOM rimette il job in fondo alla coda (max 3 tentativi). Stop: touch queue/STOP2
GPU=$1; SLOT=$2
. /mnt/20TB/asaccoia/brain2text/scripts/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while screen -ls | grep -q "b2t-w$GPU$SLOT"; do sleep 60; done
while true; do
  [ -f queue/STOP2 ] && { echo "STOP2 trovato, esco"; exit 0; }
  JOB=$(flock queue/.lock bash -c 'l=$(head -n1 queue/pending.txt 2>/dev/null); [ -n "$l" ] && { sed -i 1d queue/pending.txt; echo "$l"; }')
  if [ -z "$JOB" ]; then sleep 60; continue; fi
  LOG=logs/job_gpu${GPU}_${SLOT}_$(date +%s).log
  echo "$(date '+%F %T') [gpu$GPU.$SLOT] START $JOB  (log $LOG)" >> queue/history.txt
  CUDA_VISIBLE_DEVICES=$GPU bash -c "$JOB" > $LOG 2>&1
  RC=$?
  echo "$(date '+%F %T') [gpu$GPU.$SLOT] END rc=$RC $JOB" >> queue/history.txt
  if [ $RC -ne 0 ] && grep -q "OutOfMemoryError" $LOG; then
    N=$(grep -cF -- "OOM-RETRY $JOB" queue/history.txt)
    if [ "$N" -lt 3 ]; then
      echo "$(date '+%F %T') OOM-RETRY $JOB" >> queue/history.txt
      sleep 300
      flock queue/.lock bash -c "echo '$JOB' >> queue/pending.txt"
    fi
  fi
done
