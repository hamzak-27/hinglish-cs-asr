"""Score results/<split>/*.jsonl and print a table.  usage: python evaluate.py [split]

oWER / oTWER: official MUCS 2021 scorer (blind = leaderboard protocol; test = same rules, unofficial).
WER / CER / err_*: our normalization and code-switching breakdown (metrics.py).
"""
import json, sys
import pandas as pd
from metrics import utt_stats, summarize
from mucs import ROOT
import official

split = sys.argv[1] if len(sys.argv) > 1 else "test"
rows = []
for p in sorted((ROOT / "results" / split).glob("*.jsonl")):
    recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    res = summarize([utt_stats(r["ref"], r["hyp"]) for r in recs])
    try:
        off = official.score(recs, split)
    except Exception as e:  # scorer not available (e.g. blind set not downloaded)
        print(f"official scorer failed for {p.name}: {e}", file=sys.stderr); off = {}
    rtf = sum(r["sec"] for r in recs) / max(sum(r["dur"] for r in recs), 1e-9)
    rows.append({"run": p.stem, "n": len(recs), **off, **res, "RTF": round(rtf, 3)})
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(pd.DataFrame(rows).set_index("run"))
