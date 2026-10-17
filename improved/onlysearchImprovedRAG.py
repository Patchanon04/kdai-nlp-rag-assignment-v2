"""
ข้อ (2) ระบบ RAG ปรับปรุง — สอบถาม ค้นเอกสาร วิเคราะห์ และสรุปคำตอบ (คู่ขนานกับ Hands-on 2)

ใช้ index ที่สร้างโดย improvedRAG.py แล้วตอบชุดคำถามเดียวกับ baseline/onlysearchAuthenticRAG.py
ด้วย LLM ตัวเดียวกันและ prompt เดียวกัน (ต่างกันเฉพาะ retrieval)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from improvedRAG import ImprovedContextualRAG


def main():
    rag = ImprovedContextualRAG(use_reranker=True)

    questions = [
        'โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?',
        'อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน',
        'ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?',
        'ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?',
        'โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?'
    ]

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n\n===== คำถามที่ {i}/{len(questions)} =====\nคำถาม: {q}")
        r = rag.search_for_question(q, k=5)
        print(f"\nคำตอบ: {r['answer']}")
        results.append(r)

    out = Path(__file__).resolve().parent / "improved_rag_search_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nบันทึกผลลัพธ์ไปยัง {out} สำเร็จ")


if __name__ == "__main__":
    main()
