#!/usr/bin/env python3
"""Vast.ai GPU market scanner — logs spot prices to TSV for analysis.

Scans all GPU types every INTERVAL seconds, appends to market log.
Rotates log at 100K lines (~1 week at 10min intervals, 10 GPU types).

Usage:
    python3 market_scanner.py                    # run continuously
    python3 market_scanner.py --once             # single scan, exit
    python3 market_scanner.py --interval 300     # every 5 min
    python3 market_scanner.py --log /tmp/mkt.tsv # custom log path
"""
import argparse
import json
import os
import subprocess
import sys
import time

GPU_TYPES = [
    "RTX_3070_Ti", "RTX_3080", "RTX_3080_Ti", "RTX_3090",
    "RTX_4070", "RTX_4070_Ti",
]

DEFAULT_LOG = os.path.expanduser("~/.config/zeb/market-log.tsv")
DEFAULT_INTERVAL = 600  # 10 minutes
MAX_LINES = 100_000
ROTATE_KEEP = 50_000


def scan_gpu(gpu_name, num_gpus=1):
    """Query vastai for top offers, return list of dicts."""
    try:
        r = subprocess.run(
            ["vastai", "search", "offers",
             f"gpu_name={gpu_name} num_gpus={num_gpus}",
             "-o", "dph_total", "--raw", "--limit", "5"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode != 0:
            return []
        return json.loads(r.stdout)
    except Exception:
        return []


def scan_all(timestamp):
    """Scan all GPU types, return TSV lines."""
    lines = []
    for gpu in GPU_TYPES:
        for num_gpus in [1, 2]:
            offers = scan_gpu(gpu, num_gpus)
            for rank, o in enumerate(offers):
                mid = o.get("machine_id", "?")
                dph = o.get("dph_total", 0)
                dph_per_gpu = dph / num_gpus if num_gpus > 0 else dph
                pcie = o.get("pcie_bw", 0)
                country = o.get("geolocation", "?").replace("\t", " ")
                dlp = o.get("dlperf", 0)
                inet_down = o.get("inet_down", 0)
                lines.append(
                    f"{timestamp}\t{gpu}\t{num_gpus}\t{rank}\t"
                    f"{dph:.4f}\t{dph_per_gpu:.4f}\t{mid}\t"
                    f"{pcie:.1f}\t{dlp:.1f}\t{inet_down:.0f}\t{country}"
                )
    return lines


def rotate_log(path):
    """Keep last ROTATE_KEEP lines if over MAX_LINES."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        if len(lines) > MAX_LINES:
            # Keep header + last ROTATE_KEEP lines
            header = lines[0] if lines[0].startswith("#") else None
            keep = lines[-ROTATE_KEEP:]
            with open(path, "w") as f:
                if header:
                    f.write(header)
                f.writelines(keep)
            print(f"  rotated {len(lines)} → {len(keep)} lines", file=sys.stderr)
    except FileNotFoundError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Vast.ai GPU market scanner")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Output TSV path")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL,
                        help="Seconds between scans (default: 600)")
    parser.add_argument("--once", action="store_true", help="Single scan, then exit")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    # Write header if new file
    if not os.path.exists(args.log) or os.path.getsize(args.log) == 0:
        with open(args.log, "w") as f:
            f.write("# ts\tgpu\tnum_gpus\trank\tdph\tdph_per_gpu\tmachine_id\t"
                    "pcie_bw\tdlperf\tinet_down\tcountry\n")

    scan_num = 0
    while True:
        ts = int(time.time())
        ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        scan_num += 1

        lines = scan_all(ts)
        if lines:
            with open(args.log, "a") as f:
                for line in lines:
                    f.write(line + "\n")
            n_offers = len(lines)
            n_gpus = len(set(l.split("\t")[1] for l in lines))
            print(f"{ts_str}  scan #{scan_num}: {n_offers} offers across {n_gpus} GPU types")
        else:
            print(f"{ts_str}  scan #{scan_num}: no data (API error?)", file=sys.stderr)

        if scan_num % 10 == 0:
            rotate_log(args.log)

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
