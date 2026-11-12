import json, math, os
from datetime import datetime, timezone

IN = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
OUT = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'

MOMENTUM_CAP_H = 3.0


def parse_ts(s):
    try:
        d = datetime.fromisoformat(str(s).replace('Z', '+00:00'))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except Exception:
        return None


def ncdf(x, mu, sd):
    return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))


def main():
    with open(os.path.join(IN, 'task.json')) as f:
        task = json.load(f)
    places = int(task.get('places', 5))
    reads = task.get('reads') or []
    edges = list(task['size']['edges'])
    if edges and max(edges) > 1000:
        edges = [math.log10(e / 1e9) if e > 0 else -1e9 for e in edges]
    nbins = len(edges) - 1

    last = None
    with open(os.path.join(IN, 'history.ndjson')) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get('rows'):
                last = obj
    rows = last['rows']

    best = {}
    for r in rows:
        try:
            mc = int(r['market_cap_lamports'])
        except Exception:
            continue
        if mc <= 0:
            continue
        m = r['mint']
        if m not in best or mc > best[m]:
            best[m] = mc
    ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
    logs = [(m, math.log10(mc / 1e9)) for m, mc in ranked]
    top = logs[:60]

    # horizon in hours between last mark and last read
    mark_ts = None
    for k in ('ts', 'timestamp', 'time', 'mark', 'at', 'created_at'):
        if k in last:
            mark_ts = parse_ts(last[k])
            if mark_ts is not None:
                break
    if mark_ts is None:
        cs = [parse_ts(r.get('created_at')) for r in rows]
        cs = [c for c in cs if c is not None]
        mark_ts = max(cs) if cs else None
    read_ts = [parse_ts(x) for x in reads]
    read_ts = [x for x in read_ts if x is not None]
    if mark_ts is not None and read_ts:
        horizon_h = max(0.0, (max(read_ts) - mark_ts) / 3600.0)
    else:
        horizon_h = 1.0
    horizon_h = min(horizon_h, 24.0 * 30)

    # rank momentum (rank-shuffling spread) is damped: capped at 3h horizon
    rank_h = min(horizon_h, MOMENTUM_CAP_H)

    probs = []
    n = len(top)
    tau = 2.0 + 0.5 * math.sqrt(rank_h)
    for k in range(1, places + 1):
        ws = [math.exp(-abs(i - (k - 1)) / tau) * math.exp(-0.02 * i) for i in range(n)]
        tot = sum(ws)
        for i, (m, _) in enumerate(top):
            probs.append({'mint': m, 'place': k, 'p': ws[i] / tot})

    top5 = [m for m, _ in top[:5]]
    i = 0
    while len(top5) < 5:
        top5.append('pad%d' % i)
        i += 1

    # size sd keeps growing with the full forecast horizon
    sizes = []
    sd = 0.10 + 0.05 * math.sqrt(horizon_h)
    for k in range(1, places + 1):
        idx = min(k - 1, n - 1)
        mu = top[idx][1] if n else 0.0
        sd_k = sd * (1.0 + 0.05 * (k - 1))
        cdfs = [ncdf(e, mu, sd_k) for e in edges]
        ps = [max(0.0, cdfs[b + 1] - cdfs[b]) for b in range(nbins)]
        s = sum(ps)
        if s <= 1e-12:
            ps = [1.0 / nbins] * nbins
        else:
            ps = [p / s for p in ps]
        for b in range(nbins):
            sizes.append({'place': k, 'bin': b, 'p': ps[b]})

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'forecast.json'), 'w') as f:
        json.dump({'probabilities': probs, 'top5': top5, 'sizes': sizes}, f)


main()
