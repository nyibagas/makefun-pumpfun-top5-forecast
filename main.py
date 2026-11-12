import json
import math
import os
import pathlib
from datetime import datetime

INPUT_DIR = pathlib.Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = pathlib.Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

MICRO = 1_000_000
MOMENTUM_WINDOW_H = 1.5
MOMENTUM_SHRINK = 0.35
# pump momentum fades: never extrapolate it past this many hours
MOMENTUM_CAP_H = 3.0
# the rank kernel widens with the forecast horizon: over ten hours a place is far less sure
KERNEL_BASE = 0.18
KERNEL_PER_HOUR = 0.015
# share of each place left unnamed for coins born after the cut and for movers
NEWCOMER_BASE = 0.28
NEWCOMER_PER_PLACE = 0.03
CANDIDATES = 10
MAX_RISE = 1.0
MAX_FALL = 1.2


def parse_ts(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_marks(path):
    marks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            mark = json.loads(line)
            if mark.get("ok") and isinstance(mark.get("rows"), list):
                marks.append(mark)
    return marks


def logcap(lamports):
    return math.log10(int(lamports)) - 9.0


task = json.loads((INPUT_DIR / "task.json").read_text())
places = int(task["places"])
edges = task["size"]["edges"]
marks = load_marks(INPUT_DIR / "history.ndjson")
if not marks:
    raise SystemExit("the history holds no ok mark")

t0 = parse_ts(task["t0"])


def hours_of(mark):
    return (parse_ts(mark["at"]) - t0).total_seconds() / 3600.0


series = {}
for mark in marks:
    t = hours_of(mark)
    for row in mark["rows"]:
        cap = int(row["market_cap_lamports"])
        if cap > 0:
            series.setdefault(row["mint"], []).append((t, logcap(cap)))

last = marks[-1]
last_t = hours_of(last)
last_cap = {
    row["mint"]: logcap(row["market_cap_lamports"])
    for row in last["rows"]
    if int(row["market_cap_lamports"]) > 0
}
reads = [hours_of({"at": r}) for r in task["reads"]]
target = reads[len(reads) // 2]
horizon = max(0.0, target - last_t)


def recent_rate(mint):
    points = [(t, v) for t, v in series.get(mint, []) if t >= last_t - MOMENTUM_WINDOW_H]
    if len(points) < 2:
        return 0.0
    n = len(points)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    if den <= 0:
        return 0.0
    return max(-1.5, min(1.5, (n * sxy - sx * sy) / den))


predicted = {}
for mint, cap in last_cap.items():
    moved = cap + recent_rate(mint) * MOMENTUM_SHRINK * min(horizon, MOMENTUM_CAP_H)
    predicted[mint] = max(cap - MAX_FALL, min(cap + MAX_RISE, moved))

order = sorted(predicted.items(), key=lambda kv: (-kv[1], kv[0]))
if len(order) < places:
    raise SystemExit(f"the last mark ranks {len(order)} coins, fewer than the {places} places")

size_sd = min(0.85, max(0.35, 0.30 * max(horizon, 0.5) ** 0.45))
rank_sd = KERNEL_BASE + KERNEL_PER_HOUR * horizon


def bin_weights(mean, sd):
    out = []
    for b in range(len(edges) + 1):
        lo = -math.inf if b == 0 else edges[b - 1]
        hi = math.inf if b == len(edges) else edges[b]
        p_lo = 1.0 if lo == -math.inf else 0.5 * math.erfc((lo - mean) / (sd * math.sqrt(2)))
        p_hi = 0.0 if hi == math.inf else 0.5 * math.erfc((hi - mean) / (sd * math.sqrt(2)))
        units = int(math.floor((p_lo - p_hi) * MICRO))
        if units > 0:
            out.append({"bin": b, "p": units / MICRO})
    return out


candidates = order[: max(CANDIDATES, places)]
probabilities = []
for k in range(places):
    base = order[k][1]
    weights = [math.exp(-0.5 * ((v - base) / rank_sd) ** 2) for _, v in candidates]
    total = sum(weights)
    budget = 1.0 - min(0.80, NEWCOMER_BASE + NEWCOMER_PER_PLACE * k)
    for (mint, _), w in zip(candidates, weights):
        units = int(math.floor(w / total * budget * MICRO))
        if units > 0:
            probabilities.append({"mint": mint, "place": k + 1, "p": units / MICRO})

sizes = []
for k in range(places):
    for row in bin_weights(order[k][1], size_sd):
        sizes.append({"place": k + 1, "bin": row["bin"], "p": row["p"]})

top5 = [mint for mint, _ in order[:places]]
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / task.get("output", "forecast.json")).write_text(
    json.dumps({"probabilities": probabilities, "top5": top5, "sizes": sizes}) + "\n"
)
