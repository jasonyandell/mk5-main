#!/usr/bin/env python3
"""Find the best-value Vast.ai offer across GPU types.

Ranks by $/game/s (cost per throughput) rather than raw $/hr.
Preferred machines get a 10% value bonus from observed performance.

Usage:
    find_offer.py --max-dph 0.10 --gpus RTX_3080,RTX_4070 \
                  --blacklist 123,456 --preferred 789:3.0,101:2.6

Output (TSV): OFFER_ID  GPU_NAME  DPH  preferred|""
Exit 1 if no suitable offer found.
"""
import argparse
import json
import subprocess
import sys

# Expected games/s per GPU — from 13.9hrs of observations (Feb 2026)
EXPECTED_GPS = {
    "RTX 3060": 1.8, "RTX 3060 Ti": 2.1,
    "RTX 3070": 2.4, "RTX 3070 Ti": 2.6,
    "RTX 3080": 2.6, "RTX 3080 Ti": 2.6, "RTX 3090": 2.6,
    "RTX 4060": 1.8, "RTX 4060 Ti": 2.1,
    "RTX 4070": 2.6, "RTX 4070 Ti": 2.8,
}


def parse_preferred(s):
    """Parse 'mid1:gps1,mid2:gps2,...' into {mid: gps} dict."""
    result = {}
    if not s:
        return result
    for entry in s.split(","):
        if ":" in entry:
            mid, gps_str = entry.split(":", 1)
            try:
                result[mid] = float(gps_str)
            except ValueError:
                pass
    return result


def worst_value(fleet_tsv, exclude_iid=""):
    """Find the worst-value worker from fleet TSV (iid, label, gpu, status, dph, age, mid).

    Returns: (value, iid, label, gpu, dph, mid) or None.
    """
    worst_val, worst = -1, None
    for line in fleet_tsv.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        iid, label, gpu, _status, dph, _age, mid = (
            parts[0], parts[1], parts[2], parts[3], float(parts[4]), int(parts[5]), parts[6]
        )
        if iid == exclude_iid:
            continue
        gps = EXPECTED_GPS.get(gpu, 1.8)
        value = dph / gps
        if value > worst_val:
            worst_val = value
            worst = (value, iid, label, gpu, dph, mid)
    return worst


def main():
    parser = argparse.ArgumentParser(description="Find best-value Vast.ai GPU offer")
    parser.add_argument("--max-dph", type=float, default=0.09)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--blacklist", default="")
    parser.add_argument("--preferred", default="")
    # --worst-value: read fleet TSV from stdin, print worst-value worker
    parser.add_argument("--worst-value", action="store_true")
    parser.add_argument("--exclude-iid", default="")
    args = parser.parse_args()

    if args.worst_value:
        fleet_tsv = sys.stdin.read()
        result = worst_value(fleet_tsv, args.exclude_iid)
        if result:
            val, iid, label, gpu, dph, mid = result
            print(f"{iid}\t{label}\t{gpu}\t{dph:.3f}\t{mid}\t{val:.4f}")
        else:
            sys.exit(1)
        return

    gpus = [g for g in args.gpus.split(",") if g]  # keep underscores for vastai search
    blacklist = set(args.blacklist.split(",")) - {""}
    preferred = parse_preferred(args.preferred)

    best = None
    best_value = float("inf")

    for gpu in gpus:
        try:
            r = subprocess.run(
                ["vastai", "search", "offers", f"gpu_name={gpu} num_gpus=1",
                 "-o", "dph_total", "--raw", "--limit", "10"],
                capture_output=True, text=True, timeout=15,
            )
            if r.returncode != 0:
                continue
            offers = json.loads(r.stdout)
            for o in offers:
                mid = str(o.get("machine_id", ""))
                if mid in blacklist:
                    continue
                dph = o["dph_total"]
                if dph > args.max_dph:
                    break  # sorted by price, rest are over budget
                if mid in preferred:
                    gps = preferred[mid] * 1.1  # 10% bonus for known-good
                else:
                    gps = EXPECTED_GPS.get(o["gpu_name"], 1.8)
                value = dph / gps
                if value < best_value:
                    best = o
                    best_value = value
        except Exception:
            pass

    if best:
        mid = str(best.get("machine_id", ""))
        pref = "preferred" if mid in preferred else ""
        print(f"{best['id']}\t{best['gpu_name']}\t{best['dph_total']:.3f}\t{pref}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
