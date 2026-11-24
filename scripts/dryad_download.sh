#!/bin/bash
# Scarica tutti i file del dataset Dryad x69p8czpq in data/raw, riprendendo download parziali.
set -u
cd /mnt/20TB/asaccoia/brain2text
. ./.dryad_credentials
token() { curl -s -X POST https://datadryad.org/oauth/token -d grant_type=client_credentials --data-urlencode "client_id=$DRYAD_ID" --data-urlencode "client_secret=$DRYAD_SECRET" | python3 -c "import json,sys;print(json.load(sys.stdin)['access_token'])"; }
# ordine: prima i piccoli/essenziali
while read id name size; do
  out=data/raw/$name
  if [ -f "$out" ] && [ "$(stat -c %s "$out")" = "$size" ]; then echo "OK gia' presente $name"; continue; fi
  for try in 1 2 3 4 5; do
    T=$(token)
    echo "$(date '+%F %T') scarico $name (tentativo $try)"
    curl -sL -C - --retry 5 -H "Authorization: Bearer $T" -o "$out" "https://datadryad.org/api/v2/files/$id/download"
    [ "$(stat -c %s "$out")" = "$size" ] && { echo "$(date '+%F %T') FATTO $name"; break; }
    echo "dimensione errata per $name, riprovo"; sleep 30
  done
done < <(curl -s "https://datadryad.org/api/v2/versions/253293/files" | python3 -c "
import json,sys;fs=json.load(sys.stdin)['_embedded']['stash:files']
fs.sort(key=lambda f:(f['size']>1e8, f['path']!='competitionData.tar.gz', f['size']))
for f in fs: print(f['_links']['stash:download']['href'].split('/')[-2], f['path'], f['size'])")
echo "$(date '+%F %T') TUTTI I DOWNLOAD TERMINATI"
