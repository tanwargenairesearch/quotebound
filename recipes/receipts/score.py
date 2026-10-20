#!/usr/bin/env python3
"""Scores quotebound generate --batch outputs on the receipts recipe, and checks two outputs byte for byte.

    python3 score.py --data DIR OUT.jsonl [OUT.jsonl ...]
    python3 score.py --data DIR --same A.jsonl B.jsonl

For each OUT:
  parsed          answers that are JSON objects with the answer's keys
  values          non-null values written; not_in_receipt: those that do not stand in their receipt (stands() of
                  prepare.py, the decoder's rule), i.e. values the model made up or altered
  precision, recall, f1
                  over (field, value) pairs against gold.jsonl, each item's name, count and price counted as
                  items.name, items.count and items.price; values compared with white space collapsed and ASCII letter
                  case aside; a receipt that does not parse scores its gold values as misses
  receipts_exact  receipts whose every value and null is right (items in any order)
  timings         p50 and p95 of a row's total_ms; decode tokens a second over all rows; tokens a verification pass
                  where the rows were decoded with drafts
--same: whether every row's "text" is byte-identical in A and B, by id. Standard library only.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import ITEM, TOTALS, fold, stands, values  # noqa: E402


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def norm(v: str) -> str:
    return " ".join(fold(v).split())


def pairs(ans: dict) -> collections.Counter:
    c = collections.Counter()
    for it in ans["items"]:
        for k in ITEM:
            if it.get(k) is not None:
                c[(f"items.{k}", norm(it[k]))] += 1
    for k in TOTALS:
        if ans.get(k) is not None:
            c[(k, norm(ans[k]))] += 1
    return c


def shape_ok(a) -> bool:
    return (isinstance(a, dict) and set(a) == {"items", *TOTALS} and isinstance(a["items"], list) and
            all(isinstance(it, dict) and set(it) == set(ITEM) for it in a["items"]) and
            all(v is None or isinstance(v, str) for v in [a[k] for k in TOTALS] +
                [x for it in a["items"] for x in it.values()]))


def items_key(ans: dict):
    return sorted(tuple("" if it[k] is None else "=" + norm(it[k]) for k in ITEM) for it in ans["items"])


def pct(xs: list, q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def score(out: Path, data: Path) -> dict:
    gold = {r["id"]: r["fields"] for r in lines(data / "gold.jsonl")}
    text = {r["id"]: r["messages"][-1]["content"] for r in lines(data / "eval.jsonl")}
    rows = lines(out)
    parsed = written = invented = exact = 0
    tp = n_pred = n_gold = 0
    for r in rows:
        g = gold[r["id"]]
        n_gold += sum(pairs(g).values())
        try:
            a = json.loads(r["text"])
        except json.JSONDecodeError:
            a = None
        if not shape_ok(a):
            continue
        parsed += 1
        vs = values(a)
        written += len(vs)
        invented += sum(not stands(text[r["id"]], v) for v in vs)
        p, q = pairs(a), pairs(g)
        tp += sum((p & q).values())
        n_pred += sum(p.values())
        exact += items_key(a) == items_key(g) and all(
            (a[k] is None) == (g[k] is None) and (a[k] is None or norm(a[k]) == norm(g[k])) for k in TOTALS)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    gen = sum(r["gen_tokens"] for r in rows)
    dec = sum(r["decode_ms"] for r in rows)
    res = {"out": out.name, "rows": len(rows), "parsed": parsed, "values": written, "not_in_receipt": invented,
           "precision": round(100 * precision, 1), "recall": round(100 * recall, 1),
           "f1": round(200 * precision * recall / (precision + recall), 1) if precision + recall else 0.0,
           "receipts_exact": exact,
           "total_ms_p50": round(pct([r["total_ms"] for r in rows], 0.5), 1),
           "total_ms_p95": round(pct([r["total_ms"] for r in rows], 0.95), 1),
           "decode_tokens_per_s": round(1000 * gen / dec, 1) if dec else None,
           "truncated": sum(bool(r.get("truncated")) for r in rows)}
    if any("passes" in r for r in rows):
        res["tokens_per_pass"] = round(gen / sum(r["passes"] for r in rows), 2)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--same", action="store_true")
    ap.add_argument("outs", type=Path, nargs="+")
    a = ap.parse_args()
    if a.same:
        x, y = ({r["id"]: r["text"] for r in lines(p)} for p in a.outs[:2])
        differ = [i for i in x if x[i] != y.get(i)]
        print(json.dumps({"rows": len(x), "byte_identical": len(x) - len(differ), "differ": differ[:10],
                          "same_ids": set(x) == set(y)}))
        sys.exit(1 if differ or set(x) != set(y) else 0)
    for out in a.outs:
        print(json.dumps(score(out, a.data)))


if __name__ == "__main__":
    main()
