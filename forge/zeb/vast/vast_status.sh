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
import sys, json

instances = json.load(sys.stdin)
zeb = [i for i in instances if (i.get('label') or '').startswith('zeb-')]

if not zeb:
    print('No zeb instances running.')
    sys.exit(0)

print(f'{'Label':<22} {'GPU':<16} {'Status':<12} {'$/hr':>6}  {'ID':>10}  SSH')
print('-' * 90)

total_cost = 0
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
    print(f'{label:<22} {gpu:<16} {status:<12} {dph:>6.3f}  {iid:>10}  {ssh}')

print('-' * 90)
print(f'Total: {len(zeb)} instances, \${total_cost:.3f}/hr (\${total_cost*24:.2f}/day)')
"
fi
