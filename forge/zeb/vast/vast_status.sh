#!/usr/bin/env bash
# vast_status.sh — Show running zeb instances
#
# Usage:
#   ./vast_status.sh          # show all zeb instances
#   ./vast_status.sh --raw    # raw JSON output

set -euo pipefail

if [ "${1:-}" = "--raw" ]; then
    vastai show instances --raw | python3 -c "
import sys, json
instances = json.load(sys.stdin)
zeb = [i for i in instances if (i.get('label') or '').startswith('zeb-')]
json.dump(zeb, sys.stdout, indent=2)
print()
"
else
    vastai show instances --raw | python3 -c "
import sys, json, time

instances = json.load(sys.stdin)
zeb = [i for i in instances if (i.get('label') or '').startswith('zeb-')]

if not zeb:
    print('No zeb instances running.')
    sys.exit(0)

now = time.time()

def fmt_age(seconds):
    if seconds is None:
        return '?'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f'{h}h{m:02d}m'
    return f'{m}m{s:02d}s'

print(f\"{'Label':<22} {'GPU':<16} {'Status':<12} {'Age':>7} {'$/hr':>6}  {'ID':>10}  SSH\")
print('-' * 100)

total_cost = 0
stale = []
for i in sorted(zeb, key=lambda x: x.get('label') or ''):
    label = i.get('label') or '?'
    gpu = i.get('gpu_name') or '?'
    status = i.get('actual_status') or 'unknown'
    dph = i.get('dph_total') or 0
    iid = i.get('id') or '?'
    ssh_host = i.get('ssh_host') or ''
    ssh_port = i.get('ssh_port') or ''
    ssh = f'ssh -p {ssh_port} root@{ssh_host}' if ssh_host and ssh_port else '-'
    total_cost += dph

    # Age from start_date (epoch seconds)
    start = i.get('start_date')
    age_s = (now - start) if start else None
    age = fmt_age(age_s)

    # Flag instances stuck in loading for >10 min
    flag = ''
    if status in ('loading', 'unknown') and age_s is not None and age_s > 600:
        flag = ' *** SLOW - not started after 10min'
        stale.append((iid, label, age))

    print(f'{label:<22} {gpu:<16} {status:<12} {age:>7} {dph:>6.3f}  {iid:>10}  {ssh}{flag}')

print('-' * 100)
print(f'Total: {len(zeb)} instances, \${total_cost:.3f}/hr (\${total_cost*24:.2f}/day)')

if stale:
    print()
    print(f'WARNING: {len(stale)} instance(s) stuck loading:')
    for iid, label, age in stale:
        print(f'  vastai destroy instance {iid}   # {label}, age {age}')
"
fi
