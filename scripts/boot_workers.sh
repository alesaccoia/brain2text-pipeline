#!/bin/bash
# Avvio dei 4 worker dopo un riavvio (crontab: @reboot).
# I job iniziati e mai finiti (START senza END in queue/history.txt) tornano in testa a queue/pending.txt:
# stesso comando = stesso --name, quindi train.py riprende da runs/NAME/ckpt.pt.
# Per non far partire nulla al boot: touch queue/NOBOOT
cd /mnt/20TB/asaccoia/brain2text || exit 1
[ -f queue/NOBOOT ] && { echo "NOBOOT presente, esco"; exit 0; }
screen -ls | grep -q 'b2t-v2-g' && { echo "worker gia' attivi"; exit 0; }
for i in $(seq 30); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done

flock queue/.lock python3 - <<'EOF'
import re, time
H = 'queue/history.txt'
n = {}
order = []
for line in open(H):
    m = re.match(r'\S+ \S+ \[gpu\S+\] (START|END rc=\S+) (.*?)(  \(log .*\))?$', line.rstrip('\n'))
    if not m:
        continue
    job = m.group(2)
    if m.group(1) == 'START':
        n[job] = n.get(job, 0) + 1
        order.append(job)
    else:
        n[job] = n.get(job, 0) - 1
todo = []
for job in order:
    if n.get(job, 0) > 0 and job not in todo:
        todo.append(job)
if todo:
    rest = open('queue/pending.txt').read() if __import__('os').path.exists('queue/pending.txt') else ''
    open('queue/pending.txt', 'w').write(''.join(j + '\n' for j in todo) + rest)
    ts = time.strftime('%F %T')
    with open(H, 'a') as f:
        for j in todo:
            f.write(f'{ts} [gpu-.-] END rc=reboot {j}\n')
print('rimessi in coda:', len(todo))
EOF

for g in 0 1; do for s in 0 1; do
  screen -dmS b2t-v2-g$g$s bash -c "cd /mnt/20TB/asaccoia/brain2text && bash scripts/worker2.sh $g $s >> logs/worker2_g$g$s.log 2>&1"
done; done
echo "$(date '+%F %T') boot_workers: worker avviati" >> queue/history.txt
