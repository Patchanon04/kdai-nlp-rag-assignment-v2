"""
ข้อ (3) ประเมินเปรียบเทียบ retrieval ของ baseline (ข้อ 1) กับ improved (ข้อ 2)

ชุดทดสอบ eval/questions.json : 20 คำถามภาษาไทย แต่ละข้อมี
  - gold_doc      ไฟล์ที่มีคำตอบ
  - gold_span     ข้อความคำตอบที่คัดมาจากเอกสารแบบคำต่อคำ (ใช้ตรวจว่า chunk ที่ค้นได้ "มีคำตอบจริง")
  - gold_keywords คำสำคัญของคำตอบ
  - lexical_overlap  high = คำในคำถามตรงกับเอกสาร / low = ถามด้วยคำอื่น (ทดสอบ semantic)

ตัวชี้วัด (ไม่ต้องใช้ LLM):
  Hit@1 / Hit@3 / Hit@5  สัดส่วนคำถามที่ gold_span อยู่ใน top-1/3/5 chunk
  MRR                    1/อันดับแรกที่พบ gold_span (เฉลี่ย)
  KeywordRecall@5        สัดส่วน gold_keywords ที่ปรากฏใน top-5
  DocAcc@1               chunk อันดับ 1 มาจากไฟล์ที่ถูกต้อง
  ContextChars           ความยาว context ที่ส่งให้ LLM (เฉลี่ย)

ตัวเลือก --answers : ให้ทั้งสองระบบ generate คำตอบด้วย LLM ตัวเดียวกัน (prompt เดียวกัน) แล้วบันทึกไว้เทียบ
                     พร้อม AnswerKeywordHit = คำตอบมี gold_keyword อย่างน้อย 1 คำหรือไม่

การใช้งาน:
  python eval/evaluate.py                      # ประเมินทั้ง 3 config
  python eval/evaluate.py --answers            # + generate คำตอบ
  python eval/evaluate.py --systems baseline   # เลือกระบบ
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baseline"))
sys.path.insert(0, str(ROOT / "improved"))

RESULTS_DIR = ROOT / "eval" / "results"
CORPUS_DIR = ROOT / "corpus_input"


def norm(s):
    """ตัดช่องว่างทั้งหมด (ข้อความไทยเว้นวรรคไม่สม่ำเสมอ และไฟล์มีการตัดบรรทัดกลางประโยค)"""
    return re.sub(r"\s+", "", s or "")


def load_corpus():
    return {p.name: norm(p.read_text(encoding="utf-8")) for p in CORPUS_DIR.glob("*.md")}


def guess_source(content, corpus):
    """baseline ไม่เก็บชื่อไฟล์ใน index -> หาไฟล์ที่มีข้อความ chunk นี้"""
    probe = norm(content)[50:250] or norm(content)[:200]
    for name, text in corpus.items():
        if probe and probe in text:
            return name
    return "?"


def make_system(name, shared):
    import llm_client  # noqa
    if name == "baseline":
        from onlysearchAuthenticRAG import AuthenticSearchRAG
        return AuthenticSearchRAG()
    from improvedRAG import ImprovedContextualRAG
    if name == "improved-norerank":
        return ImprovedContextualRAG(use_reranker=False, embed_model=shared.get("embed"))
    if name == "improved":
        return ImprovedContextualRAG(use_reranker=True, embed_model=shared.get("embed"))
    raise ValueError(name)


def evaluate_system(name, rag, questions, corpus, k=5, answers=False):
    rows = []
    for q in questions:
        t0 = time.time()
        results = rag.retrieve(q["question"], k=k)
        latency = time.time() - t0
        for r in results:
            r.setdefault("source", guess_source(r["content"], corpus))
        span = norm(q["gold_span"])
        ranks = [i for i, r in enumerate(results, 1) if span in norm(r["content"])]
        first = ranks[0] if ranks else None
        top5_text = norm(" ".join(r["content"] for r in results[:5]))
        kw_hit = [kw for kw in q["gold_keywords"] if norm(kw) in top5_text]
        context = rag.build_context(results)
        row = {
            "id": q["id"], "question": q["question"], "gold_doc": q["gold_doc"],
            "lexical_overlap": q.get("lexical_overlap", "high"),
            "first_rank": first,
            "hit@1": int(first == 1), "hit@3": int(bool(first) and first <= 3), "hit@5": int(bool(first) and first <= 5),
            "rr": (1.0 / first) if first else 0.0,
            "keyword_recall@5": len(kw_hit) / max(1, len(q["gold_keywords"])),
            "doc_acc@1": int(bool(results) and results[0].get("source") == q["gold_doc"]),
            "context_chars": len(context), "retrieval_sec": round(latency, 3),
            "top": [{"rank": i, "source": r.get("source"), "score": round(float(r["score"]), 4),
                     "breadcrumb": r.get("breadcrumb", ""), "preview": r["content"][:120].replace("\n", " ")}
                    for i, r in enumerate(results, 1)],
        }
        if answers:
            ans = rag.generate_response(q["question"], context)
            row["answer"] = ans
            row["answer_keyword_hit"] = int(any(norm(kw) in norm(ans) for kw in q["gold_keywords"]))
        rows.append(row)
        mark = "✓" if first else "✗"
        print(f"  {q['id']} {mark} rank={first} kw={row['keyword_recall@5']:.2f} doc={row['doc_acc@1']} | {q['question'][:50]}")
    return rows


def summarize(rows, answers=False):
    def mean(key, subset):
        return sum(r[key] for r in subset) / max(1, len(subset))
    keys = ["hit@1", "hit@3", "hit@5", "rr", "keyword_recall@5", "doc_acc@1", "context_chars", "retrieval_sec"]
    if answers:
        keys.append("answer_keyword_hit")
    out = {"n": len(rows)}
    for kname in keys:
        out[kname if kname != "rr" else "mrr"] = round(mean(kname, rows), 3)
    for grp in ("high", "low"):
        sub = [r for r in rows if r["lexical_overlap"] == grp]
        out[f"hit@5_{grp}_overlap"] = round(mean("hit@5", sub), 3)
        out[f"mrr_{grp}_overlap"] = round(mean("rr", sub), 3)
    return out


def write_summary_md(summaries, answers):
    cols = [("hit@1", "Hit@1"), ("hit@3", "Hit@3"), ("hit@5", "Hit@5"), ("mrr", "MRR"),
            ("keyword_recall@5", "KeywordRecall@5"), ("doc_acc@1", "DocAcc@1"),
            ("hit@5_high_overlap", "Hit@5 (คำตรง)"), ("hit@5_low_overlap", "Hit@5 (คำไม่ตรง)"),
            ("context_chars", "Context (ตัวอักษร)"), ("retrieval_sec", "เวลา/คำถาม (s)")]
    if answers:
        cols.append(("answer_keyword_hit", "AnswerKeywordHit"))
    lines = ["| ระบบ | " + " | ".join(c[1] for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(str(s.get(c[0], "")) for c in cols) + " |")
    md = "\n".join(lines) + "\n"
    (RESULTS_DIR / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default="baseline,improved-norerank,improved")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--answers", action="store_true", help="generate คำตอบด้วย LLM ด้วย")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    corpus = load_corpus()
    shared = {}
    summaries = {}
    for name in args.systems.split(","):
        print(f"\n===== {name} =====")
        rag = make_system(name, shared)
        if hasattr(rag, "embed_model") and "embed" not in shared:
            shared["embed"] = rag.embed_model
        rows = evaluate_system(name, rag, questions, corpus, k=args.k, answers=args.answers)
        summaries[name] = summarize(rows, args.answers)
        (RESULTS_DIR / f"{name}.json").write_text(
            json.dumps({"summary": summaries[name], "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summaries[name], ensure_ascii=False))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(summaries, args.answers)


if __name__ == "__main__":
    main()
