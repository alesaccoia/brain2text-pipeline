#!/bin/bash
# Aspetta che lm_decoder sia compilato, poi avvia il servizio di decodifica 5-gram (pruned, ~50 GB RAM)
cd /mnt/20TB/asaccoia/brain2text
until .venv-lm/bin/python -c "import lm_decoder" 2>/dev/null; do
  screen -ls | grep -q b2t-build || { echo "build terminata senza lm_decoder: vedi logs/build_lm.log"; exit 1; }
  sleep 120
done
echo "$(date) lm_decoder pronto, avvio decodifica"
exec .venv-lm/bin/python -u src/decode_ngram.py --lm 5gram
