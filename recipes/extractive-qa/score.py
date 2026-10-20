#!/usr/bin/env python3
"""Scores quotebound generate --batch outputs on the extractive-QA recipe, and checks two outputs byte for byte.

    python3 score.py --data DIR OUT.jsonl [OUT.jsonl ...]
    python3 score.py --data DIR --same A.jsonl B.jsonl

For each OUT: rows, answers that parse as {"answer": str}, answers that stand in their passage as whole words
(letter case aside), SQuAD exact match and F1 against gold.jsonl (the official normalisation: lower case, no
punctuation, no articles), and timings (p50 and p95 of a row's total_ms; decode tokens a second, all rows'
gen_tokens over all decode_ms; tokens a verification pass where the rows were decoded with drafts).
--same: whether every row's "text" is byte-identical in A and B, by id. Standard library only.
"""
import argparse
import collections
import json
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare import stands  # noqa: E402


def lines(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def normalize(s: str) -> str:
    s = "".join(c for c in s.lower() if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if not p or not g:
        return float(p == g)
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def pct(xs: list, q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def score(out: Path, data: Path) -> dict:
    gold = {r["id"]: r["answers"] for r in lines(data / "gold.jsonl")}
    passage = {r["id"]: r["messages"][-1]["content"] for r in lines(data / "eval.jsonl")}
    rows = lines(out)
    parsed = standing = 0
    em = f = 0.0
    for r in rows:
        try:
            ans = json.loads(r["text"])["answer"]
            assert isinstance(ans, str)
            parsed += 1
        except Exception:
            continue
        standing += stands(passage[r["id"]], ans)
        em += max(float(normalize(ans) == normalize(g)) for g in gold[r["id"]])
        f += max(f1(ans, g) for g in gold[r["id"]])
    n = len(rows)
    gen = sum(r["gen_tokens"] for r in rows)
    dec = sum(r["decode_ms"] for r in rows)
    res = {"out": out.name, "rows": n, "parsed": parsed, "stands_in_passage": standing,
           "exact_match": round(100 * em / n, 1), "f1": round(100 * f / n, 1),
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
