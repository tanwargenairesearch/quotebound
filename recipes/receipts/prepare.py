#!/usr/bin/env python3
"""CORD v2 receipts as quotebound training and evaluation files: a receipt's items and totals as JSON, every value
copied from the receipt.

    python3 prepare.py --out DIR

Reads the ground_truth column of CORD v2 (https://huggingface.co/datasets/naver-clova-ix/cord-v2, CC BY 4.0) at a
pinned revision, through HTTP range reads, so the receipt images (2.3 GB) are not downloaded. Needs pyarrow and
huggingface_hub. Writes:
  train.jsonl  CORD's train split, {"id", "messages": [system, user, assistant]}
  dev.jsonl    CORD's validation split (the trainer's dev loss), the same form
  eval.jsonl   CORD's test split, {"id", "messages": [system, user]}, for generate --batch
  gold.jsonl   {"id", "fields"} for eval.jsonl: the labelled values, as answer() writes them
The user turn is the receipt's labelled words, a line per printed row in reading order. The assistant turn is the
answer in compact JSON (json.dumps with ensure_ascii=False and no spaces), which is what --schema schema.json
--quote-source user lets the decoder write. A train or dev receipt is kept only when every value stands in its text
as the decoder requires (stands() below) and no field holds two values. A test receipt is kept when no field holds two
values, even when a value does not stand in its text: such a value counts as a miss.
"""
import argparse
import collections
import json
from pathlib import Path

REPO = "datasets/naver-clova-ix/cord-v2"
REVISION = "7f0115a4b758a71d6473b8d085751692da2fef98"
FILES = {
    "train": ["train-00000-of-00004-b4aaeceff1d90ecb", "train-00001-of-00004-7dbbe248962764c5",
              "train-00002-of-00004-688fe1305a55e5cc", "train-00003-of-00004-2d0cd200555ed7fd"],
    "validation": ["validation-00000-of-00001-cc3c5779fe22e8ca"],
    "test": ["test-00000-of-00001-9c204eb3f4e11791"],
}

SYSTEM = ("Read the receipt and return its items and totals as JSON. Copy every value exactly as the receipt prints "
          "it, and use null for a value the receipt does not show.")

# The answer's fields and the CORD labels they come from.
TOTALS = {"subtotal": ("sub_total", "subtotal_price"), "tax": ("sub_total", "tax_price"),
          "service": ("sub_total", "service_price"), "discount": ("sub_total", "discount_price"),
          "total": ("total", "total_price"), "cash": ("total", "cashprice"), "change": ("total", "changeprice"),
          "card": ("total", "creditcardprice")}
ITEM = {"name": "nm", "count": "cnt", "price": "price"}


# The decoder's quote rule (src/constrain.cpp, quote_text): the quote occurs in the text, ASCII letter case aside,
# with no white space at either end, not inside a word, and not inside a wider number.
def word(c: str) -> bool:
    return c.isascii() and (c.isalnum() or c == "_")


def digit(c: str) -> bool:
    return "0" <= c <= "9"


def fold(s: str) -> str:
    return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in s)


def stands(text: str, quote: str) -> bool:
    if not quote:
        return False
    t, q = fold(text), fold(quote)
    i = t.find(q)
    while i >= 0:
        j = i + len(q)
        left = i == 0 or not ((word(t[i - 1]) and word(t[i])) or (digit(t[i]) and t[i - 1] in ",.£"))
        right = j == len(t) or not ((word(t[j]) and word(t[j - 1])) or
                                    (digit(t[j - 1]) and (t[j] == "%" or (t[j] in ",." and j + 1 < len(t)
                                                                          and digit(t[j + 1])))))
        if left and right and not t[i].isspace() and not t[j - 1].isspace():
            return True
        i = t.find(q, i + 1)
    return False


def receipt_text(gt: dict) -> str:
    """The labelled words, grouped by CORD's row ids, rows top to bottom and words left to right."""
    rows = collections.defaultdict(list)
    for line in gt["valid_line"]:
        for w in line["words"]:
            q = w["quad"]
            rows[w["row_id"]].append((min(q["y1"], q["y2"]), q["x1"], w["text"]))
    ordered = sorted(rows.values(), key=lambda ws: sum(w[0] for w in ws) / len(ws))
    return "\n".join(" ".join(t for _, _, t in sorted(ws, key=lambda w: w[1])) for ws in ordered)


def fields(gt: dict):
    """The answer as a dict, or None when a field holds more than one value or an item has no name."""
    parse = gt["gt_parse"]
    menu = parse.get("menu", [])
    out = {"items": []}
    for it in menu if isinstance(menu, list) else [menu]:
        item = {k: it.get(v) for k, v in ITEM.items()}
        if not isinstance(item["name"], str) or any(v is not None and not isinstance(v, str) for v in item.values()):
            return None
        out["items"].append(item)
    for k, (group, label) in TOTALS.items():
        v = parse.get(group, {}).get(label) if isinstance(parse.get(group), dict) else None
        if v is not None and not isinstance(v, str):
            return None
        out[k] = v
    return out


def values(answer: dict) -> list:
    return [v for it in answer["items"] for v in it.values() if v is not None] + \
           [answer[k] for k in TOTALS if answer[k] is not None]


def answer_json(answer: dict) -> str:
    return json.dumps(answer, ensure_ascii=False, separators=(",", ":"))


def read_split(split: str) -> list:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    gts = []
    for name in FILES[split]:
        with fs.open(f"{REPO}@{REVISION}/data/{name}.parquet", "rb", block_size=1 << 20) as f:
            gts += [json.loads(x) for x in pq.ParquetFile(f).read(columns=["ground_truth"])
                    .column("ground_truth").to_pylist()]
    return gts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}

    def write(name, items):
        with open(a.out / name, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    for split, name in (("train", "train.jsonl"), ("validation", "dev.jsonl")):
        rows, dropped = [], collections.Counter()
        for gt in read_split(split):
            text, ans = receipt_text(gt), fields(gt)
            if ans is None:
                dropped["two values in a field, or an item without a name"] += 1
            elif not all(stands(text, v) for v in values(ans)):
                dropped["a value the decoder cannot quote"] += 1
            else:
                rows.append({"id": f"{split}-{gt['meta']['image_id']}",
                             "messages": [{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": "Receipt:\n" + text},
                                          {"role": "assistant", "content": answer_json(ans)}]})
        write(name, rows)
        report[split] = {"kept": len(rows), "dropped": dict(dropped)}

    evals, gold, skipped = [], [], 0
    for gt in read_split("test"):
        ans = fields(gt)
        if ans is None:
            skipped += 1
            continue
        rid = f"test-{gt['meta']['image_id']}"
        evals.append({"id": rid, "messages": [{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": "Receipt:\n" + receipt_text(gt)}]})
        gold.append({"id": rid, "fields": ans})
    write("eval.jsonl", evals)
    write("gold.jsonl", gold)
    unquotable = sum(not stands(e["messages"][1]["content"], v) for e, g in zip(evals, gold) for v in values(g["fields"]))
    report["test"] = {"kept": len(evals), "skipped_two_values": skipped, "gold_values": sum(len(values(g["fields"]))
                      for g in gold), "gold_values_not_quotable": unquotable}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
