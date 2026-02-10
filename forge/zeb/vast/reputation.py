#!/usr/bin/env python3
"""Persistent machine reputation scores for Vast.ai fleet management.

Tracks per-machine: observed gps, boot reliability, boot times, stalls, errors.
Used by find_offer.py and the monitor for value-based fleet decisions.

Usage:
    # Update from monitor (JSON lines on stdin):
    echo '{"mid":"123","gpu":"RTX 4070","event":"ok","gps":2.6}' | python3 reputation.py update

    # Query effective gps for offer ranking:
    python3 reputation.py query 123 456 789

    # Dump all machine scores:
    python3 reputation.py dump

Events: ok, stall, error, boot_success (with boot_time), boot_fail
"""
import json
import os
import sys
import time

REP_FILE = os.path.expanduser("~/.config/zeb/machine-reputation.json")

# How many gps samples before we trust observed over expected
MIN_SAMPLES = 3

EXPECTED_GPS = {
    "RTX 3060": 1.8, "RTX 3060 Ti": 2.1,
    "RTX 3070": 2.4, "RTX 3070 Ti": 2.6,
    "RTX 3080": 2.6, "RTX 3080 Ti": 2.6, "RTX 3090": 2.6,
    "RTX 4060": 1.8, "RTX 4060 Ti": 2.1,
    "RTX 4070": 2.6, "RTX 4070 Ti": 2.8,
}


def load():
    try:
        with open(REP_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(data):
    os.makedirs(os.path.dirname(REP_FILE), exist_ok=True)
    tmp = REP_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, REP_FILE)


def update_machine(rep, mid, gpu, event, **kwargs):
    """Update reputation for a single event."""
    if mid not in rep:
        rep[mid] = {
            "gpu": gpu, "gps_sum": 0, "gps_count": 0,
            "boots": 0, "boot_ok": 0,
            "boot_time_sum": 0, "boot_time_count": 0,
            "stalls": 0, "errors": 0, "last_seen": 0,
        }
    m = rep[mid]
    m["gpu"] = gpu or m.get("gpu", "?")
    m["last_seen"] = int(time.time())

    if event == "ok":
        gps = kwargs.get("gps")
        if gps and gps > 0:
            m["gps_sum"] = m.get("gps_sum", 0) + gps
            m["gps_count"] = m.get("gps_count", 0) + 1
    elif event == "stall":
        m["stalls"] = m.get("stalls", 0) + 1
    elif event == "error":
        m["errors"] = m.get("errors", 0) + 1
    elif event == "boot_success":
        m["boot_ok"] = m.get("boot_ok", 0) + 1
        boot_time = kwargs.get("boot_time")
        if boot_time and boot_time > 0:
            m["boot_time_sum"] = m.get("boot_time_sum", 0) + boot_time
            m["boot_time_count"] = m.get("boot_time_count", 0) + 1
    elif event == "boot_fail":
        pass  # boots is incremented on boot_start
    elif event == "boot_start":
        m["boots"] = m.get("boots", 0) + 1


def effective_gps(m):
    """Compute effective gps: observed avg * reliability."""
    gpu = m.get("gpu", "?")
    count = m.get("gps_count", 0)
    if count >= MIN_SAMPLES:
        gps_avg = m["gps_sum"] / count
    else:
        gps_avg = EXPECTED_GPS.get(gpu, 1.8)

    boots = m.get("boots", 0)
    if boots >= MIN_SAMPLES:
        reliability = m.get("boot_ok", 0) / boots
        reliability = max(reliability, 0.1)  # floor at 10%
    else:
        reliability = 1.0

    return gps_avg * reliability


def avg_boot_time(m):
    """Average boot time in seconds, or None."""
    count = m.get("boot_time_count", 0)
    if count > 0:
        return m["boot_time_sum"] / count
    return None


def cmd_update():
    """Read JSON events from stdin, update reputation."""
    rep = load()
    count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = str(evt.get("mid", ""))
        if not mid or mid == "?":
            continue
        update_machine(
            rep, mid,
            gpu=evt.get("gpu", ""),
            event=evt.get("event", ""),
            gps=evt.get("gps"),
            boot_time=evt.get("boot_time"),
        )
        count += 1
    save(rep)
    print(f"updated {count} events", file=sys.stderr)


def cmd_query(machine_ids):
    """Print effective gps for given machine IDs (pipe-delimited output)."""
    rep = load()
    parts = []
    for mid in machine_ids:
        m = rep.get(mid)
        if m:
            parts.append(f"{mid}:{effective_gps(m):.2f}")
    print("|".join(parts))


def cmd_dump():
    """Human-readable dump of all machines."""
    rep = load()
    if not rep:
        print("No reputation data yet.")
        return

    print(f"{'MID':>8}  {'GPU':<16}  {'gps_avg':>7}  {'eff_gps':>7}  "
          f"{'boots':>5}  {'ok%':>5}  {'boot_t':>6}  {'stalls':>6}  {'errors':>6}")
    print("-" * 90)

    machines = sorted(rep.items(), key=lambda x: effective_gps(x[1]), reverse=True)
    for mid, m in machines:
        count = m.get("gps_count", 0)
        gps_avg = m["gps_sum"] / count if count > 0 else 0
        eff = effective_gps(m)
        boots = m.get("boots", 0)
        ok_pct = (m.get("boot_ok", 0) / boots * 100) if boots > 0 else 0
        bt = avg_boot_time(m)
        bt_str = f"{bt:.0f}s" if bt else "-"
        print(f"{mid:>8}  {m.get('gpu', '?'):<16}  {gps_avg:>7.2f}  {eff:>7.2f}  "
              f"{boots:>5}  {ok_pct:>4.0f}%  {bt_str:>6}  "
              f"{m.get('stalls', 0):>6}  {m.get('errors', 0):>6}")


def main():
    if len(sys.argv) < 2:
        print("Usage: reputation.py {update|query|dump}", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "update":
        cmd_update()
    elif cmd == "query":
        cmd_query(sys.argv[2:])
    elif cmd == "dump":
        cmd_dump()
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
