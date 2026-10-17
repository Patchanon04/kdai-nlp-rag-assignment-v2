"""
ข้อ (1) ระบบ RAG พื้นฐาน (Baseline) — Hands-on 2: สอบถาม ค้นเอกสาร วิเคราะห์ และสรุปคำตอบ (สไลด์หน้า 25)

โค้ดนี้คือ onlysearchAuthenticRAG.py จาก https://github.com/aekanun2020/2025-authenticRAG
คงตรรกะการค้นหาทั้งหมดตามต้นฉบับ:
  - sparse_search : BM25 (multi_match บน content + contextualized_content)
  - dense_search  : k-NN cosine บน embedding ของ bge-m3
  - hybrid_search : รวมอันดับด้วย Reciprocal Rank Fusion (k=60) แล้วตัด top-k
  - generate_response : ส่ง context (content + context ของแต่ละ chunk) ให้ LLM ตอบ

สิ่งเดียวที่แก้: การเรียก LLM ย้ายไป llm_client.py (ตัวเดียวกับระบบปรับปรุง)
"""
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import llm_client

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


class AuthenticSearchRAG:
    def __init__(self, opensearch_host="localhost", opensearch_port=9200):
        print(f"LLM: {llm_client.describe()}")

        # ตั้งค่า embedding model
        self.embed_model = SentenceTransformer('BAAI/bge-m3')
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        # สร้าง wrapper สำหรับ embedding
        class EmbeddingsWrapper:
            def __init__(self, embed_model):
                self.embed_model = embed_model

            def embed_query(self, text):
                return self.embed_model.encode(text).tolist()

        self.encoder = EmbeddingsWrapper(self.embed_model)

        # ตั้งค่าการเชื่อมต่อกับ OpenSearch
        self.opensearch_client = OpenSearch(
            hosts=[{'host': opensearch_host, 'port': opensearch_port}],
            use_ssl=False
        )

        # ชื่อ indices ใน OpenSearch (จาก authenticRAG.py)
        self.vector_index_name = "anthropic-vector-index"
        self.bm25_index_name = "anthropic-bm25-index"

        # ค่า k สำหรับ RRF
        self.rrf_k = 60

    def call_qwen_api(self, prompt, max_tokens=500, temperature=0.2):
        """เรียกใช้ LLM เพื่อสร้างข้อความ (ชื่อเมธอดคงไว้ตามต้นฉบับ)"""
        return llm_client.call_llm(prompt, max_tokens=max_tokens, temperature=temperature)

    def sparse_search(self, query, k=10):
        """ค้นหาด้วย BM25 (คล้ายตัวอย่าง)"""
        response = self.opensearch_client.search(
            index=self.bm25_index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "contextualized_content"],
                        "type": "best_fields"
                    }
                },
                "size": k
            }
        )

        return [(hit["_id"], hit["_score"], hit["_source"]) for hit in response["hits"]["hits"]]

    def dense_search(self, query, k=10):
        """ค้นหาด้วย vector similarity"""
        query_embedding = self.encoder.embed_query(query)

        response = self.opensearch_client.search(
            index=self.vector_index_name,
            body={
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": k
                        }
                    }
                },
                "size": k
            }
        )

        return [(hit["_id"], hit["_score"], hit["_source"]) for hit in response["hits"]["hits"]]

    def rrf_fusion(self, rankings, k=60):
        """ผสมผลลัพธ์การค้นหาด้วย Reciprocal Rank Fusion (เหมือนในตัวอย่าง)"""
        fused_scores = {}
        doc_sources = {}

        for ranker_id, results in enumerate(rankings):
            for rank, (doc_id, score, source) in enumerate(results):
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_sources[doc_id] = source

                # RRF formula: 1 / (k + rank)
                fused_scores[doc_id] += 1.0 / (k + rank)

        # เรียงลำดับตาม RRF score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score, doc_sources[doc_id]) for doc_id, score in sorted_results]

    def hybrid_search(self, query, k=5, rrf_k=60):
        """ค้นหาแบบ hybrid ด้วย RRF ที่รวมผลจาก sparse และ dense search (เหมือนตัวอย่าง)"""
        # ค้นหาด้วย BM25 (sparse)
        sparse_results = self.sparse_search(query, k=k*2)
        print(f"\nSparse search found {len(sparse_results)} results")

        # ค้นหาด้วย vector similarity (dense)
        dense_results = self.dense_search(query, k=k*2)
        print(f"Dense search found {len(dense_results)} results")

        # ใช้ RRF ผสมผลลัพธ์
        rankings = [sparse_results, dense_results]
        fused_results = self.rrf_fusion(rankings, k=rrf_k)
        print(f"RRF fusion returned {len(fused_results)} results")

        return fused_results[:k]

    def truncate_context(self, context, max_tokens=4000):
        """ย่อ context ให้มีขนาดไม่เกิน max_tokens"""
        words = context.split()
        estimated_tokens = len(words) * 1.3

        if estimated_tokens <= max_tokens:
            return context

        target_words = int(max_tokens / 1.3)
        truncated_context = " ".join(words[:target_words])
        return truncated_context + " ...(truncated)"

    def retrieve(self, question, k=5):
        """ค้นหาอย่างเดียว (ไม่เรียก LLM) — ใช้โดย eval/evaluate.py"""
        results = self.hybrid_search(question, k=k)
        return [{"doc_id": doc_id, "score": score,
                 "content": doc.get("content", ""),
                 "context": doc.get("contextualized_content", "")}
                for doc_id, score, doc in results]

    def build_context(self, search_results):
        """ประกอบ context string จากผลค้นหา (รูปแบบเดียวกับต้นฉบับ)"""
        context = ""
        for r in search_results:
            context += f"Content: {r['content']}\nContext: {r['context']}\n\n"
        return self.truncate_context(context)

    def search_for_question(self, question, k=5):
        """ค้นหาเอกสารที่เกี่ยวข้องกับคำถามและสร้างคำตอบ"""
        results = self.hybrid_search(question, k=k)

        if not results:
            return {"question": question, "answer": "ไม่พบข้อมูลที่เกี่ยวข้อง", "results": []}

        # แสดงเอกสารที่พบและใช้เนื้อหาเป็น context
        context = ""
        print(f"\nFound {len(results)} relevant documents for question: '{question}'")

        search_results = []
        for i, (doc_id, score, doc) in enumerate(results, 1):
            print(f"\nDOCUMENT {i}/{len(results)}")
            print(f"Score: {score}")
            print(f"Doc ID: {doc_id}")

            # เพิ่มข้อมูลเอกสารลงในผลลัพธ์
            result_info = {
                "doc_id": doc_id,
                "score": score,
                "content": doc.get("content", ""),
                "context": doc.get("contextualized_content", "")
            }
            search_results.append(result_info)

            # แสดงบริบทที่สร้างขึ้น
            if "contextualized_content" in doc:
                print(f"Context: {doc['contextualized_content']}")

            # แสดงเนื้อหาเฉพาะเอกสารแรก
            if i == 1:
                print("\nContent:")
                content = doc.get("content", "")
                if len(content) > 500:
                    print(content[:500] + "...(truncated)")
                else:
                    print(content)

            # เพิ่มเนื้อหาและบริบทเข้าไปใน context
            content = doc.get("content", "")
            context_info = doc.get("contextualized_content", "")
            context += f"Content: {content}\nContext: {context_info}\n\n"

        # ย่อ context ถ้าจำเป็น
        truncated_context = self.truncate_context(context)

        # สร้างคำตอบจาก context
        answer = self.generate_response(question, truncated_context)
        return {"question": question, "answer": answer, "results": search_results}

    def generate_response(self, question, context="", max_tokens=1000):
        """สร้างคำตอบจากคำถามและ context"""
        if not context:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Question: {question}"""
        else:
            prompt = f"""You are a question answering assistant. Answer the question as truthful and helpful as possible.

Context information:
{context}

Question: {question}"""

        return self.call_qwen_api(prompt, max_tokens=max_tokens, temperature=0.6)

    def search_multiple_questions(self, questions, k=5):
        """ค้นหาและตอบหลายคำถาม"""
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n\n===== คำถามที่ {i}/{len(questions)} =====")
            print(f"คำถาม: {question}")

            try:
                # ค้นหาและตอบคำถาม
                result = self.search_for_question(question, k=k)

                # เพิ่มผลลัพธ์
                results.append(result)

                # แสดงคำตอบ
                print(f"\nคำตอบ: {result['answer']}")
            except Exception as e:
                print(f"เกิดข้อผิดพลาดในการค้นหาคำถาม '{question}': {e}")
                results.append({"question": question, "answer": f"เกิดข้อผิดพลาด: {str(e)}", "results": []})

        return results

    def export_results_to_json(self, results, output_file="search_results.json"):
        """ส่งออกผลลัพธ์ไปยังไฟล์ JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nบันทึกผลลัพธ์ไปยัง {output_file} สำเร็จ")


def main():
    # ตั้งค่าระบบ Search
    rag = AuthenticSearchRAG(
        opensearch_host="localhost",
        opensearch_port=9200
    )

    # ชุดคำถามที่ต้องการค้นหา (ตามต้นฉบับ)
    questions = [
        'โรคหัดและโรคหัดเยอรมันแตกต่างกันอย่างไร?',
        'อธิบายสาเหตุของโรคหัดเยอรมันและการป้องกัน',
        'ทำไมโรคหัดเยอรมันจึงมีอันตรายกับหญิงตั้งครรภ์?',
        'ถ้าคนที่ฉีดวัคซีนป้องกันโรคหัดเยอรมันแล้ว จะมีโอกาสติดเชื้อหรือไม่?',
        'โรคหัดเยอรมันมีผลกระทบอย่างไรต่อระบบสาธารณสุขและเศรษฐกิจของประเทศ?'
    ]

    # ค้นหาและตอบคำถาม
    results = rag.search_multiple_questions(questions, k=5)

    # ส่งออกผลลัพธ์ไปยังไฟล์ JSON
    out = Path(__file__).resolve().parent / "authentic_rag_search_results.json"
    rag.export_results_to_json(results, str(out))


if __name__ == "__main__":
    main()
