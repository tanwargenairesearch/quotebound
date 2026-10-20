#!/usr/bin/env python3
"""SQuAD v1.1 as quotebound training and evaluation files: answer a question with the passage's own words.

    python3 prepare.py --squad DIR --out DIR [--train-n 2000] [--dev-n 200] [--eval-n 300] [--seed 0]

DIR holds train-v1.1.json and dev-v1.1.json (https://rajpurkar.github.io/SQuAD-explorer/, CC BY-SA 4.0). Writes:
  train.jsonl  --train-n questions from SQuAD's training articles, {"messages": [system, user, assistant]}
  dev.jsonl    --dev-n questions from other training articles (the trainer's dev loss), the same form
  eval.jsonl   --eval-n questions from SQuAD's dev set, {"id", "messages": [system, user]}, for generate --batch
  gold.jsonl   {"id", "answers"} for eval.jsonl, SQuAD's reference answers
The assistant turn is {"answer": "<words of the passage>"} as json.dumps(ensure_ascii=False) writes it compactly, which is what
--schema schema.json --quote-source user lets the decoder write. A training question is kept only when its first
reference answer stands in the passage as whole words (letter case aside), as the decoder's quote rule requires.
Standard library only.
"""
import argparse
import json
import random
import re
from pathlib import Path

SYSTEM = ("Answer the question from the passage. Reply with JSON {\"answer\": ...}, where the answer is the shortest "
          "stretch of the passage's own words that answers the question.")


def user_turn(context: str, question: str) -> str:
    return f"Passage:\n{context.strip()}\n\nQuestion: {question.strip()}"


def word(c: str) -> bool:
    return c.isascii() and (c.isalnum() or c == "_")


def stands(text: str, quote: str) -> bool:
    """quote occurs in text, letter case aside, not inside a word and with no white space at either end."""
    if not quote or quote != quote.strip():
        return False
    t, q = text.lower(), quote.lower()
    for m in re.finditer(re.escape(q), t):
        i, j = m.start(), m.end()
        if (i == 0 or not (word(t[i - 1]) and word(t[i]))) and (j == len(t) or not (word(t[j - 1]) and word(t[j]))):
            return True
    return False


def questions(path: Path):
    """(article index, context, qa) for every question, in file order."""
    for a, article in enumerate(json.loads(path.read_text())["data"]):
        for p in article["paragraphs"]:
            for qa in p["qas"]:
                yield a, p["context"], qa


def answer_json(answer: str) -> str:
    return json.dumps({"answer": answer}, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--squad", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--train-n", type=int, default=2000)
    ap.add_argument("--dev-n", type=int, default=200)
    ap.add_argument("--eval-n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    train_all = list(questions(a.squad / "train-v1.1.json"))
    articles = sorted({art for art, _, _ in train_all})
    rng.shuffle(articles)
    dev_articles = set(articles[: max(1, len(articles) // 10)])  # a tenth of the articles never trained on

    def rows(pool, n):
        pool = [x for x in pool if stands(x[1], x[2]["answers"][0]["text"])]
        rng.shuffle(pool)
        return [{"messages": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user_turn(ctx, qa["question"])},
                              {"role": "assistant", "content": answer_json(qa["answers"][0]["text"])}],
                 "id": qa["id"]} for _, ctx, qa in pool[:n]]

    train = rows([x for x in train_all if x[0] not in dev_articles], a.train_n)
    dev = rows([x for x in train_all if x[0] in dev_articles], a.dev_n)
    evals = list(questions(a.squad / "dev-v1.1.json"))
    rng.shuffle(evals)
    evals = evals[: a.eval_n]

    a.out.mkdir(parents=True, exist_ok=True)

    def write(name, items):
        with open(a.out / name, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    write("train.jsonl", train)
    write("dev.jsonl", dev)
    write("eval.jsonl", [{"id": qa["id"], "messages": [{"role": "system", "content": SYSTEM},
                                                       {"role": "user", "content": user_turn(ctx, qa["question"])}]}
                         for _, ctx, qa in evals])
    write("gold.jsonl", [{"id": qa["id"], "answers": [x["text"] for x in qa["answers"]]} for _, _, qa in evals])
    print(json.dumps({"train": len(train), "dev": len(dev), "eval": len(evals), "dev_articles": len(dev_articles),
                      "seed": a.seed}))


if __name__ == "__main__":
    main()
