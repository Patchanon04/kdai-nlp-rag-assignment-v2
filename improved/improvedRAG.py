"""
ข้อ (2) ระบบ RAG ปรับปรุง (Improved) — นำเข้าเอกสาร + ค้นหา

ต่อยอดจาก baseline (ข้อ 1) โดย "ไม่เปลี่ยน LLM ที่ใช้ generate คำตอบสุดท้าย" (ใช้ llm_client.py ตัวเดียวกัน
และ prompt ตอบคำถามประโยคเดียวกันกับ baseline) สิ่งที่เปลี่ยนคือส่วน retrieval ทั้งหมด:

  [C] Chunking
      - baseline: MarkdownNodeParser(chunk_size=256) ซึ่งจริง ๆ ไม่สน chunk_size — แบ่งตามหัวข้อ (#) เท่านั้น
        ได้ 53 chunk ขนาด 112–8,404 ตัวอักษร (กลาง ๆ ~1,200) → chunk ใหญ่/เล็กไม่สม่ำเสมอ embedding เจือจาง
      - improved: แยกตามหัวข้อก่อน แล้วแบ่งย่อยเป็น chunk ≤ CHUNK_MAX_CHARS ตามย่อหน้า/ประโยค
        (ตัดประโยคไทยด้วย pythainlp crfcut) มี overlap 1 ประโยค, ตัดส่วน boilerplate (เอกสารอ้างอิง ฯลฯ)
        และเก็บ "breadcrumb" ของหัวข้อ (เรื่อง > หัวข้อ > หัวข้อย่อย) ติดไปกับทุก chunk

  [X] Contextual retrieval (แก้บั๊กของ baseline)
      - baseline: ตั้งใจส่ง "เอกสารทั้งฉบับ + chunk" ให้ LLM สร้างบริบท แต่โค้ดส่ง chunk ซ้ำแทนเอกสาร
        (full_doc_content = doc.page_content) บริบทที่ได้จึงเป็นแค่การสรุป chunk ตัวเอง และมักเป็นภาษาอังกฤษ
      - improved: ส่งเอกสารเต็มจริง ๆ สร้างบริบทระดับ "หัวข้อ" (section) 1 ครั้ง ให้ทุก chunk ย่อยในหัวข้อนั้นใช้ร่วมกัน
        (ประหยัดการเรียก LLM) และบังคับให้บริบทเป็นภาษาไทย/ระบุชื่อโรค เพื่อให้ BM25 ไทยและ embedding ใช้ได้

  [E] Embedding
      - ใช้ BAAI/bge-m3 ตัวเดิม แต่ embed ข้อความ = breadcrumb + บริบทหัวข้อ + เนื้อหา chunk และ normalize เวกเตอร์
        (cosine ที่ถูกต้อง) ใช้ HNSW index แบบระบุ method ชัดเจน

  [S] Search algorithm / scoring
      - BM25: baseline ใช้ standard analyzer ซึ่งไม่ตัดคำไทย — ข้อความไทยทั้งวลีกลายเป็น token เดียว
        (ทดสอบด้วย _analyze: "ระยะฟักตัวของโรคหัดเยอรมันนานกี่วัน" → 1 token) BM25 จึงแทบไม่ทำงานกับภาษาไทย
        improved ตัดคำไทยด้วย pythainlp (newmm) ก่อน index และก่อน query + ใช้ OpenSearch `thai` analyzer เป็นฟิลด์เสริม
      - ขยาย candidate pool (k*4 ต่อ retriever) แล้วรวมด้วย weighted RRF

  [R] Reranking (ใหม่)
      - cross-encoder BAAI/bge-reranker-v2-m3 (รองรับไทย) ให้คะแนน (คำถาม, chunk) โดยตรง แล้วเลือก top-k สุดท้าย

ผลลัพธ์เชิงตัวเลขดูใน eval/evaluate.py และ README.md
"""
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import llm_client

from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer, CrossEncoder
from pythainlp.tokenize import word_tokenize, sent_tokenize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus_input"
CONTEXT_CACHE_FILE = Path(__file__).resolve().parent / "context_cache.json"

# หัวข้อที่เป็น boilerplate ของเว็บต้นทาง — ไม่ใช่เนื้อหาความรู้ ตัดทิ้ง
SKIP_SECTION_PREFIXES = ("เอกสารอ้างอิง", "เรื่องที่เกี่ยวข้อง", "เมดไทย")

CHUNK_MAX_CHARS = 700     # ขนาด chunk สูงสุด (ตัวอักษร)
CHUNK_MIN_CHARS = 150     # chunk เล็กกว่านี้จะถูกรวมกับ chunk ก่อนหน้า


def load_reranker():
    """cross-encoder สำหรับ rerank — ใช้ GPU (MPS บน Mac) ถ้ามี และจำกัดความยาว 512 token (เร็วขึ้น ~12 เท่า)"""
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    return CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)


def thai_tokens(text):
    """ตัดคำไทยด้วย pythainlp newmm คืน string ที่คั่นด้วยช่องว่าง (ให้ whitespace analyzer ของ OpenSearch ใช้)"""
    toks = [t.strip().lower() for t in word_tokenize(text, engine="newmm", keep_whitespace=False)]
    return " ".join(t for t in toks if t)


# ----------------------------------------------------------------------------------------------
# 1) Chunking
# ----------------------------------------------------------------------------------------------
def parse_markdown_sections(path):
    """อ่าน Markdown -> list ของ section {source, breadcrumb, header, text}
    - ติดตาม header stack (#, ##, ###) เพื่อสร้าง breadcrumb
    - ตัดบรรทัด '---' (page break จากต้นทาง) และ byline 'โดย เมดไทย...'
    - ข้าม section ที่เป็น boilerplate"""
    source = Path(path).name
    raw = Path(path).read_text(encoding="utf-8")
    sections = []
    stack = []            # [(level, title)]
    buf = []

    def flush():
        text = "\n".join(buf).strip()
        buf.clear()
        if not text or not stack:
            return
        title = stack[-1][1]
        if title.strip().startswith(SKIP_SECTION_PREFIXES):
            return
        sections.append({
            "source": source,
            "breadcrumb": " > ".join(t for _, t in stack),
            "header": title,
            "text": text,
        })

    for line in raw.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            continue
        if line.strip() == "---" or line.startswith("โดย เมดไทย"):
            continue
        buf.append(line.rstrip())
    flush()
    return sections


def split_into_paragraphs(text):
    paras, cur = [], []
    for line in text.splitlines():
        if line.strip():
            cur.append(line.strip())
        elif cur:
            paras.append("\n".join(cur)); cur = []
    if cur:
        paras.append("\n".join(cur))
    return paras


def split_long_paragraph(para, max_chars):
    """ย่อหน้ายาวเกิน -> แบ่งตามประโยคไทย (crfcut) แล้วแพ็คให้ ≤ max_chars พร้อม overlap 1 ประโยค"""
    sents = [s for s in sent_tokenize(para, engine="crfcut") if s.strip()]
    if not sents:
        return [para]
    chunks, cur = [], []
    for s in sents:
        if cur and len("".join(cur)) + len(s) > max_chars:
            chunks.append("".join(cur))
            cur = [cur[-1]] if len(cur[-1]) < max_chars // 2 else []   # overlap ประโยคสุดท้าย
        cur.append(s)
    if cur:
        chunks.append("".join(cur))
    return chunks


def chunk_section(section, max_chars=CHUNK_MAX_CHARS, min_chars=CHUNK_MIN_CHARS):
    """แพ็คย่อหน้าของ section ให้เป็น chunk ≤ max_chars (ย่อหน้ายาวแบ่งตามประโยค)"""
    pieces = []
    for p in split_into_paragraphs(section["text"]):
        pieces.extend(split_long_paragraph(p, max_chars) if len(p) > max_chars else [p])

    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur); cur = p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur:
        if chunks and len(cur) < min_chars:
            chunks[-1] = chunks[-1] + "\n" + cur
        else:
            chunks.append(cur)
    return chunks


# ----------------------------------------------------------------------------------------------
# 2) ระบบ RAG
# ----------------------------------------------------------------------------------------------
class ImprovedContextualRAG:
    def __init__(self, opensearch_host="localhost", opensearch_port=9200,
                 use_reranker=True, rerank_pool=15, rrf_k=60, w_sparse=1.0, w_dense=1.0,
                 embed_model=None):
        print(f"LLM: {llm_client.describe()}")
        self.embed_model = embed_model or SentenceTransformer("BAAI/bge-m3")
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()
        self.use_reranker = use_reranker
        self.reranker = load_reranker() if use_reranker else None
        self.rerank_pool = rerank_pool
        self.rrf_k = rrf_k
        self.w_sparse, self.w_dense = w_sparse, w_dense

        self.client = OpenSearch(hosts=[{"host": opensearch_host, "port": opensearch_port}], use_ssl=False)
        self.vector_index_name = "improved-vector-index"
        self.bm25_index_name = "improved-bm25-index"
        self._create_indices()

    # ---------- indices ----------
    def _create_indices(self):
        if not self.client.indices.exists(index=self.vector_index_name):
            self.client.indices.create(index=self.vector_index_name, body={
                "settings": {"index.knn": True},
                "mappings": {"properties": {
                    "embedding": {"type": "knn_vector", "dimension": self.embedding_dim,
                                  "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}},
                    "doc_id": {"type": "keyword"}, "source": {"type": "keyword"},
                    "breadcrumb": {"type": "keyword"}, "content": {"type": "text"},
                    "contextualized_content": {"type": "text"}, "chunk_id": {"type": "integer"},
                }}})
            print(f"สร้าง vector index {self.vector_index_name} สำเร็จ")
        if not self.client.indices.exists(index=self.bm25_index_name):
            self.client.indices.create(index=self.bm25_index_name, body={
                "settings": {
                    "analysis": {"analyzer": {
                        # ฟิลด์ *_tok ถูกตัดคำด้วย pythainlp มาแล้ว -> แค่แยกตามช่องว่าง + lowercase
                        "pretokenized": {"type": "custom", "tokenizer": "whitespace", "filter": ["lowercase"]},
                    }},
                    "similarity": {"default": {"type": "BM25"}},
                },
                "mappings": {"properties": {
                    "content": {"type": "text", "analyzer": "thai"},
                    "content_tok": {"type": "text", "analyzer": "pretokenized"},
                    "contextualized_content": {"type": "text", "analyzer": "thai"},
                    "contextualized_content_tok": {"type": "text", "analyzer": "pretokenized"},
                    "doc_id": {"type": "keyword"}, "source": {"type": "keyword"},
                    "breadcrumb": {"type": "keyword"}, "chunk_id": {"type": "integer"},
                }}})
            print(f"สร้าง BM25 index {self.bm25_index_name} สำเร็จ")

    def delete_indices(self):
        for idx in (self.vector_index_name, self.bm25_index_name):
            if self.client.indices.exists(index=idx):
                self.client.indices.delete(index=idx)
        self._create_indices()

    # ---------- contextual retrieval ----------
    @staticmethod
    def get_context_prompt(document_content, section_content):
        # prompt ของ Anthropic เหมือน baseline + บังคับภาษาไทยและชื่อโรค (เพื่อให้ BM25/embedding ไทยใช้ได้)
        return f"""<document>
{document_content}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{section_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
Write the context in Thai, 1-2 sentences, and always name the disease the document is about.
"""

    def _load_cache(self):
        if CONTEXT_CACHE_FILE.exists():
            return json.loads(CONTEXT_CACHE_FILE.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self, cache):
        CONTEXT_CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    def generate_section_contexts(self, sections, full_docs, workers=4):
        """สร้างบริบท 1 ครั้งต่อ section (ขนานกัน) พร้อม cache ลงไฟล์ เพื่อ re-index ได้โดยไม่เรียก LLM ซ้ำ"""
        cache = self._load_cache()

        def key(sec):
            return hashlib.sha1(f"{sec['source']}|{sec['breadcrumb']}|{sec['text']}".encode()).hexdigest()

        todo = [s for s in sections if key(s) not in cache]
        print(f"สร้างบริบทระดับหัวข้อ: {len(sections)} section (ต้องเรียก LLM {len(todo)} ครั้ง, cache {len(sections)-len(todo)})")

        def work(sec):
            prompt = self.get_context_prompt(full_docs[sec["source"]], f"{sec['breadcrumb']}\n{sec['text']}")
            out = llm_client.call_llm(prompt, max_tokens=120, temperature=0.1).strip()
            if out.startswith("LLM error"):
                raise RuntimeError(out)
            return sec, out

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (sec, ctx) in enumerate(ex.map(work, todo), 1):
                cache[key(sec)] = ctx
                print(f"  [{i}/{len(todo)}] {sec['source']} | {sec['header'][:40]} -> {ctx[:60]}...")
                if i % 5 == 0:
                    self._save_cache(cache)
        self._save_cache(cache)
        if todo:
            print(f"  ใช้เวลา {time.time()-t0:.1f}s")
        for s in sections:
            s["context"] = cache[key(s)]
        return sections

    # ---------- indexing ----------
    def build_chunks(self, md_paths):
        full_docs, sections = {}, []
        for p in md_paths:
            full_docs[Path(p).name] = Path(p).read_text(encoding="utf-8")
            sections.extend(parse_markdown_sections(p))
        self.generate_section_contexts(sections, full_docs)

        chunks = []
        for sec in sections:
            for body in chunk_section(sec):
                chunks.append({
                    "chunk_id": len(chunks), "source": sec["source"], "breadcrumb": sec["breadcrumb"],
                    "content": body,
                    # บริบท = breadcrumb + บริบทหัวข้อจาก LLM (เก็บแยกจากเนื้อหาเหมือน baseline)
                    "contextualized_content": f"{sec['breadcrumb']}\n{sec['context']}",
                })
        sizes = [len(c["content"]) for c in chunks]
        print(f"แบ่งได้ {len(chunks)} chunk จาก {len(sections)} section | ขนาด min/median/max = "
              f"{min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)} ตัวอักษร")
        return chunks

    def index_documents(self, md_paths, reset=True):
        if reset:
            self.delete_indices()
        chunks = self.build_chunks(md_paths)

        # embedding ของ (บริบท + เนื้อหา) แบบ normalize (batch)
        texts = [f"{c['contextualized_content']}\n\n{c['content']}" for c in chunks]
        t0 = time.time()
        embs = self.embed_model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
        print(f"embed {len(texts)} chunk ใช้เวลา {time.time()-t0:.1f}s")

        vec_actions, bm25_actions = [], []
        for c, e in zip(chunks, embs):
            _id = f"doc_{c['chunk_id']}"
            vec_actions.append({"_index": self.vector_index_name, "_id": _id, "_source": {
                "embedding": e.tolist(), "doc_id": _id, "chunk_id": c["chunk_id"], "source": c["source"],
                "breadcrumb": c["breadcrumb"], "content": c["content"],
                "contextualized_content": c["contextualized_content"]}})
            bm25_actions.append({"_index": self.bm25_index_name, "_id": _id, "_source": {
                "doc_id": _id, "chunk_id": c["chunk_id"], "source": c["source"], "breadcrumb": c["breadcrumb"],
                "content": c["content"], "content_tok": thai_tokens(c["content"]),
                "contextualized_content": c["contextualized_content"],
                "contextualized_content_tok": thai_tokens(c["contextualized_content"])}})
        helpers.bulk(self.client, vec_actions)
        helpers.bulk(self.client, bm25_actions)
        self.client.indices.refresh(index=self.vector_index_name)
        self.client.indices.refresh(index=self.bm25_index_name)
        print(f"index {len(chunks)} chunk ลง OpenSearch สำเร็จ")
        return len(chunks)

    # ---------- search ----------
    def sparse_search(self, query, k=10):
        """BM25 บนฟิลด์ที่ตัดคำไทยแล้ว (น้ำหนักหลัก) + ฟิลด์ thai analyzer (เสริม)"""
        q_tok = thai_tokens(query)
        resp = self.client.search(index=self.bm25_index_name, body={
            "query": {"bool": {"should": [
                {"multi_match": {"query": q_tok, "fields": ["content_tok^2", "contextualized_content_tok"],
                                 "type": "best_fields"}},
                {"multi_match": {"query": query, "fields": ["content", "contextualized_content"],
                                 "type": "best_fields"}},
            ]}},
            "size": k, "_source": {"excludes": ["content_tok", "contextualized_content_tok"]}})
        return [(h["_id"], h["_score"], h["_source"]) for h in resp["hits"]["hits"]]

    def dense_search(self, query, k=10):
        qe = self.embed_model.encode(query, normalize_embeddings=True).tolist()
        resp = self.client.search(index=self.vector_index_name, body={
            "query": {"knn": {"embedding": {"vector": qe, "k": k}}},
            "size": k, "_source": {"excludes": ["embedding"]}})
        return [(h["_id"], h["_score"], h["_source"]) for h in resp["hits"]["hits"]]

    def rrf_fusion(self, rankings, weights, k=60):
        """weighted Reciprocal Rank Fusion: score = Σ w_i / (k + rank_i)"""
        fused, sources = {}, {}
        for results, w in zip(rankings, weights):
            for rank, (doc_id, _score, src) in enumerate(results):
                fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank)
                sources.setdefault(doc_id, src)
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [(d, s, sources[d]) for d, s in ordered]

    def rerank(self, query, candidates, k):
        """cross-encoder ให้คะแนน (query, breadcrumb + content) ตรง ๆ แล้วจัดอันดับใหม่"""
        pairs = [(query, f"{src['breadcrumb']}\n{src['content']}") for _, _, src in candidates]
        scores = self.reranker.predict(pairs, batch_size=16)
        ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        return [(d, float(s), src) for (d, _rrf, src), s in ranked[:k]]

    def hybrid_search(self, query, k=5, verbose=False):
        pool = k * 4
        sparse = self.sparse_search(query, k=pool)
        dense = self.dense_search(query, k=pool)
        fused = self.rrf_fusion([sparse, dense], [self.w_sparse, self.w_dense], k=self.rrf_k)
        if verbose:
            print(f"\nSparse {len(sparse)} | Dense {len(dense)} | RRF {len(fused)} candidates")
        if self.use_reranker:
            return self.rerank(query, fused[:self.rerank_pool], k)
        return fused[:k]

    def retrieve(self, question, k=5):
        """ค้นหาอย่างเดียว (ไม่เรียก LLM) — ใช้โดย eval/evaluate.py"""
        return [{"doc_id": d, "score": s, "source": src["source"], "breadcrumb": src["breadcrumb"],
                 "content": src["content"], "context": src["contextualized_content"]}
                for d, s, src in self.hybrid_search(question, k=k)]

    @staticmethod
    def build_context(results):
        ctx = ""
        for r in results:
            ctx += f"[แหล่งที่มา: {r['source']} | {r['breadcrumb']}]\nContent: {r['content']}\nContext: {r['context']}\n\n"
        return ctx

    # prompt ตอบคำถาม "เหมือน baseline ทุกตัวอักษร" และ LLM ตัวเดียวกัน — ต่างกันแค่ context ที่ค้นมาได้
    def generate_response(self, question, context="", max_tokens=1000):
        if not context:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Question: {question}"""
        else:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Context information:
{context}

Question: {question}"""
        return llm_client.call_llm(prompt, max_tokens=max_tokens, temperature=0.6)

    def search_for_question(self, question, k=5):
        results = self.retrieve(question, k=k)
        if not results:
            return {"question": question, "answer": "ไม่พบข้อมูลที่เกี่ยวข้อง", "results": []}
        print(f"\nFound {len(results)} chunks for: '{question}'")
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['source']} | {r['breadcrumb'][:70]}")
        answer = self.generate_response(question, self.build_context(results))
        return {"question": question, "answer": answer, "results": results}


def main():
    print(f"LLM check: {llm_client.check_ready()}")
    md_paths = [str(CORPUS_DIR / n) for n in ["1.md", "2.md", "44.md", "5555.md"]]
    rag = ImprovedContextualRAG(use_reranker=False)   # ตอน index ไม่ต้องโหลด reranker
    rag.index_documents(md_paths, reset=True)

    question = "ผมจะรู้ได้อย่างไรว่าผมเป็นโรคหัดแบบไหน"
    rag.use_reranker, rag.reranker = True, load_reranker()
    res = rag.search_for_question(question)
    print(f"\nQuestion: {question}\nFinal Answer: {res['answer']}")


if __name__ == "__main__":
    main()
