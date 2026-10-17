# KDAI-NLP Assignment — Authentic (Contextual) RAG: Baseline vs Improved

งานนี้ทำตาม assignment 5 ข้อของวิชา (แทน quiz) โดยอ้างอิงสไลด์หน้า 22 (Hands-on 1: นำเข้าเอกสาร) และหน้า 25
(Hands-on 2: สอบถาม ค้นเอกสาร วิเคราะห์ และสรุปคำตอบ) และโค้ดจาก
[aekanun2020/2025-authenticRAG](https://github.com/aekanun2020/2025-authenticRAG)

| ข้อ | สิ่งที่ส่ง | อยู่ที่ |
|---|---|---|
| (1) สร้างระบบ RAG ตามสไลด์ 22, 25 | ระบบ baseline | [`baseline/authenticRAG.py`](baseline/authenticRAG.py), [`baseline/onlysearchAuthenticRAG.py`](baseline/onlysearchAuthenticRAG.py) |
| (2) ปรับปรุง chunking / embedding / search / scoring / reranking (ไม่เปลี่ยน LLM ตอบคำถาม) | ระบบ improved | [`improved/improvedRAG.py`](improved/improvedRAG.py), [`improved/onlysearchImprovedRAG.py`](improved/onlysearchImprovedRAG.py) |
| (3) อธิบายความแตกต่าง ดีขึ้นอย่างไร เพราะเหตุใด | หัวข้อ "ข้อ (3)" ด้านล่าง + ผลการวัด | [`eval/`](eval/) |

---

## โครงสร้างโปรเจกต์

```
kdai-nlp-rag-assignment/
├── corpus_input/              # เอกสาร Markdown 4 ไฟล์ (หัดเยอรมัน, อหิวาตกโรค, ต้อกระจก, กรดไหลย้อน) จาก repo ของวิชา
├── llm_client.py              # ตัวกลางเรียก LLM "ตัวเดียวกัน" ให้ทั้งสองระบบ (Claude CLI หรือ OpenRouter/Qwen)
├── baseline/                  # ข้อ (1)
│   ├── authenticRAG.py        #   Hands-on 1: chunk -> สร้าง context -> embed -> index (vector + BM25)
│   └── onlysearchAuthenticRAG.py  # Hands-on 2: hybrid search (BM25 + vector, RRF) -> LLM ตอบ
├── improved/                  # ข้อ (2)
│   ├── improvedRAG.py         #   chunking ใหม่, contextual retrieval ที่แก้แล้ว, BM25 ตัดคำไทย, weighted RRF, reranker
│   ├── onlysearchImprovedRAG.py
│   └── context_cache.json     #   บริบทระดับหัวข้อที่ LLM สร้าง (cache ไว้ให้ re-index ได้โดยไม่เรียก LLM ซ้ำ)
├── eval/                      # ข้อ (3)
│   ├── questions.json         #   ชุดคำถามทดสอบ 20 ข้อ พร้อมคำตอบอ้างอิง (gold span) ที่คัดจากเอกสาร
│   ├── evaluate.py            #   วัด Hit@k, MRR, KeywordRecall, DocAcc และ generate คำตอบเทียบกัน
│   └── results/               #   ผลลัพธ์ (summary.md, *.json รายข้อ, คำตอบของทั้งสองระบบ)
├── docker-compose.yml         # OpenSearch 2.19.1
└── requirements.txt
```

## การติดตั้งและรัน

```bash
# 1) OpenSearch (ตาม readme ของวิชา)
docker compose up -d

# 2) Python 3.10+ (ทดสอบด้วย 3.11)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) เลือก LLM (ใช้ตัวเดียวกันทั้ง baseline และ improved)
#    ค่า default: Claude ผ่าน Claude Code CLI (`claude -p`) — ต้อง login CLI ไว้ก่อน
export LLM_PROVIDER=claude-cli   # LLM_MODEL=sonnet (default)
#    หรือใช้ OpenRouter + Qwen เหมือนต้นฉบับของวิชา:
# export LLM_PROVIDER=openai OPENROUTER_API_KEY=sk-or-...   # LLM_MODEL=qwen/qwen-2.5-72b-instruct (default)

# 4) ข้อ (1) baseline: นำเข้าเอกสาร แล้วถาม-ตอบ
python baseline/authenticRAG.py
python baseline/onlysearchAuthenticRAG.py

# 5) ข้อ (2) improved: นำเข้าเอกสาร แล้วถาม-ตอบ (ชุดคำถามเดียวกัน)
python improved/improvedRAG.py
python improved/onlysearchImprovedRAG.py

# 6) ข้อ (3) ประเมินเปรียบเทียบ (retrieval อย่างเดียว / เพิ่ม --answers เพื่อ generate คำตอบด้วย)
python eval/evaluate.py --answers
```

> การเรียก LLM ของต้นฉบับ (`call_qwen_api` → OpenAI client → OpenRouter) ถูกย้ายไปรวมที่ `llm_client.py`
> เพื่อรับประกันว่า **LLM ที่ generate คำตอบสุดท้ายของทั้งสองระบบเป็นตัวเดียวกันและใช้ prompt เดียวกันทุกตัวอักษร**
> ตรรกะอื่นของ baseline คงไว้ตามต้นฉบับทั้งหมด

---

## ข้อ (1) ระบบ RAG พื้นฐาน (baseline) ตามสไลด์หน้า 22 และ 25

เป็น "Anthropic-style Contextual RAG" ตามโค้ดของวิชา

**Hands-on 1 — นำเข้าเอกสาร (`baseline/authenticRAG.py`)**
1. อ่าน Markdown ด้วย `TextLoader` → แปลงเป็น LlamaIndex `Document`
2. แบ่ง chunk ด้วย `MarkdownNodeParser(chunk_size=256)`
3. ทุก chunk เรียก LLM สร้าง "บริบท" ด้วย prompt ของ Anthropic (`<document>…</document>` + `<chunk>…</chunk>`)
4. embed `chunk + บริบท` ด้วย `BAAI/bge-m3` → เก็บใน `anthropic-vector-index` (k-NN, cosine)
5. เก็บ `content` และ `contextualized_content` แยกฟิลด์ใน `anthropic-bm25-index` (standard analyzer, BM25)

**Hands-on 2 — ถาม ค้น วิเคราะห์ สรุป (`baseline/onlysearchAuthenticRAG.py`)**
1. `sparse_search` BM25 (`multi_match` บน content + contextualized_content) top 2k
2. `dense_search` k-NN บน embedding ของคำถาม top 2k
3. `rrf_fusion` รวมอันดับด้วย Reciprocal Rank Fusion (k=60) → top k=5
4. ประกอบ `Content: … Context: …` ของทั้ง 5 chunk เป็น context แล้วให้ LLM ตอบ → export JSON

---

## ข้อ (2) ระบบ RAG ปรับปรุง (improved)

เปลี่ยนเฉพาะฝั่ง retrieval; LLM และ prompt ตอบคำถามเหมือน baseline ทุกประการ

| ส่วน | baseline (ข้อ 1) | improved (ข้อ 2) |
|---|---|---|
| **Chunking** | `MarkdownNodeParser(chunk_size=256)` — จริง ๆ parser นี้ไม่สน `chunk_size` เลย แบ่งตามหัวข้อ `#` อย่างเดียว → 53 chunk ขนาด 112–8,404 ตัวอักษร (median 1,188) | แบ่งตามหัวข้อก่อน แล้วแบ่งย่อยตามย่อหน้า/ประโยคไทย (`pythainlp` crfcut) ให้ ≤ 700 ตัวอักษร, overlap 1 ประโยค, รวม chunk จิ๋วเข้ากับก่อนหน้า, ตัด boilerplate (เอกสารอ้างอิง/เรื่องที่เกี่ยวข้อง/byline) → 159 chunk ขนาด 57–739 (median 545) + เก็บ **breadcrumb** หัวข้อ (เรื่อง > หัวข้อ > หัวข้อย่อย) ติดทุก chunk |
| **Contextual retrieval** | ตั้งใจส่ง "เอกสารทั้งฉบับ" ให้ LLM แต่โค้ดส่ง chunk ซ้ำ (`full_doc_content = doc.page_content`) → บริบทเป็นแค่สรุป chunk ตัวเอง ไม่รู้ว่าอยู่ส่วนไหนของเรื่องอะไร และมักออกมาเป็นภาษาอังกฤษ | ส่งเอกสารเต็มจริง, สร้างบริบท **ระดับหัวข้อ** 1 ครั้ง (40 ครั้งแทน 159) ให้ chunk ย่อยใช้ร่วมกัน, บังคับเป็นภาษาไทยและระบุชื่อโรค, บริบทที่เก็บ = breadcrumb + บริบทจาก LLM |
| **Embedding** | `bge-m3` embed `chunk + บริบท` ทีละ chunk ไม่ normalize | `bge-m3` ตัวเดิม embed `breadcrumb + บริบทหัวข้อ + chunk` แบบ batch, `normalize_embeddings=True` (cosine ถูกต้อง), HNSW ระบุ method ชัดเจน |
| **BM25 / tokenization** | `standard` analyzer — **ไม่ตัดคำไทย** ข้อความไทยทั้งวลีกลายเป็น token เดียว BM25 จึงจับได้แค่คำอังกฤษ/ตัวเลข/วลีที่พิมพ์ตรงเป๊ะ | ตัดคำไทยด้วย `pythainlp` (newmm) ทั้งตอน index และตอน query (ฟิลด์ `*_tok`, น้ำหนัก 2 เท่า) + ฟิลด์เสริมด้วย OpenSearch `thai` analyzer |
| **Search / scoring** | sparse top 2k + dense top 2k → RRF (k=60) → top k | sparse top 4k + dense top 4k → **weighted RRF** → 15 candidate |
| **Reranking** | ไม่มี | **cross-encoder `BAAI/bge-reranker-v2-m3`** ให้คะแนน (คำถาม, breadcrumb + chunk) โดยตรง → top k=5 |
| **Context ที่ส่งให้ LLM** | 5 หัวข้อเต็ม ๆ (เฉลี่ย ~11,000 ตัวอักษร) | 5 chunk ที่เจาะจง + ระบุแหล่งที่มา/หัวข้อ (เฉลี่ย ~3,600 ตัวอักษร) |

หลักฐานว่า BM25 ของ baseline ไม่ตัดคำไทย (ทดสอบกับ OpenSearch 2.19.1 `_analyze`):

```
standard : ["ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน", "เฉลี่ยเท่าไร", "mmr", "vaccine"]
thai     : ["ระยะ", "ฟักตัว", "โรค", "หัดเยอรมัน", "นาน", "กี่", "เฉลี่ย", "เท่าไร", "mmr", "vaccine"]
newmm    : ["ระยะ", "ฟักตัว", "ของ", "โรคหัดเยอรมัน", "นาน", "กี่", "วัน", "เฉลี่ย", "เท่าไร", "mmr"]
```

---

## ข้อ (3) ต่างกันตรงไหน ดีขึ้นอย่างไร เพราะเหตุใด

### วิธีวัด

`eval/questions.json` มีคำถามภาษาไทย 20 ข้อ (หัดเยอรมัน 6, อหิวาตกโรค 5, ต้อกระจก 4, กรดไหลย้อน 5) แต่ละข้อมี
`gold_span` = ข้อความคำตอบที่คัดมาจากเอกสาร **คำต่อคำ** จึงตรวจอัตโนมัติได้ว่า chunk ที่ค้นได้ "มีคำตอบจริง" หรือไม่
คำถาม 9 ข้อตั้งใจใช้คำที่ **ไม่ตรง** กับเอกสาร (เช่น "ทำแท้ง" ↔ "ยุติการตั้งครรภ์", "กักตัว" ↔ "ห้ามออกนอกบ้าน",
"ยกหัวเตียง" ↔ "หนุนศีรษะ") เพื่อทดสอบการค้นเชิงความหมาย และ 4 ข้อเป็นหัวข้อที่มีในทุกโรค (ภาวะแทรกซ้อน/การป้องกัน)
เพื่อทดสอบว่าแยกโรคถูกไหม

ตัวชี้วัด: **Hit@k** (gold span อยู่ใน top-k), **MRR**, **KeywordRecall@5**, **DocAcc@1** (chunk อันดับ 1 มาจากไฟล์ถูก),
ขนาด context ที่ส่งให้ LLM และ **AnswerKeywordHit** (คำตอบที่ LLM generate มีคำสำคัญของคำตอบจริง)
ทั้งสองระบบใช้ LLM ตัวเดียวกัน prompt เดียวกัน จึงเทียบได้ว่าผลต่างมาจาก retrieval ล้วน ๆ

### ผลการวัด

<!-- RESULTS -->

### อธิบายผล: ดีขึ้นตรงไหน เพราะอะไร

<!-- ANALYSIS -->

### ข้อจำกัด / สิ่งที่ยังไม่ได้ทำ

- ชุดทดสอบมี 20 ข้อ และ gold span ถูกคัดโดยผู้ทำงานเอง จึงเป็นการประเมินเชิงเปรียบเทียบ ไม่ใช่ benchmark มาตรฐาน
- Hit@k ของ baseline ได้เปรียบเชิงโครงสร้าง เพราะ chunk ใหญ่มาก (ทั้งหัวข้อ) → แค่เจอหัวข้อถูกก็ "มีคำตอบ" แต่ LLM ต้องอ่าน context ยาวกว่า ~3 เท่า
- reranker ทำให้ช้าลง (~4 วินาที/คำถาม บน Apple M-series ด้วย MPS) แลกกับอันดับที่แม่นขึ้น
- ยังไม่ได้ทำ query rewriting / HyDE / fine-tune embedding ซึ่งเป็นแนวทางต่อยอดได้

---

## หมายเหตุเรื่อง LLM

- โค้ดต้นฉบับใช้ `qwen/qwen-2.5-72b-instruct` ผ่าน OpenRouter; โปรเจกต์นี้ยังรองรับทางนั้น (`LLM_PROVIDER=openai`)
- การทดลองในรายงานนี้รันด้วย Claude ผ่าน Claude Code CLI (`LLM_PROVIDER=claude-cli`) ทั้งตอนสร้างบริบทและตอนตอบคำถาม
  **ของทั้งสองระบบเหมือนกัน** จึงไม่ขัดกับเงื่อนไขข้อ (2) ที่ห้ามเปลี่ยน LLM ตอบคำถามสุดท้ายระหว่าง (1) กับ (2)
