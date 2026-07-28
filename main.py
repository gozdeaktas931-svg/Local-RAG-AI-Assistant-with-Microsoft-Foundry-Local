import json
import os
import sqlite3
import time
from typing import List, Tuple

from foundry_local_sdk import Configuration, FoundryLocalManager
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer

# ====================================================================
# CONFIGURATION SETTINGS
# ====================================================================
DOCUMENT_FILE = "dokuman.pdf"
DATABASE_FILE = "rag_memory.db"
EMBED_MODEL_IDENTIFIER = "BAAI/bge-small-en-v1.5"
TARGET_LLM = "qwen2.5-0.5b"

CHUNK_SIZE_LIMIT = 400
CHUNK_OVERLAP_SIZE = 40
RETRIEVAL_LIMIT = 3


class LocalRAGAssistant:

    def __init__(self, pdf_path: str, db_path: str):
        self.pdf_path = pdf_path
        self.db_path = db_path
        self.embedder = None
        self.chat_client = None

    def prepare_vector_store(self):
        """PDF metinlerini işler, filtreler ve vektörleştirerek SQLite'a kaydeder."""
        print("[+] PDF belgesi yükleniyor...")
        if not os.path.exists(self.pdf_path):
            print(
                f"[ERR] Belirtilen dosya yolda bulunamadı: '{self.pdf_path}'"
            )
            exit(1)

        pdf_loader = PyPDFLoader(self.pdf_path)
        raw_docs = pdf_loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE_LIMIT, chunk_overlap=CHUNK_OVERLAP_SIZE
        )
        parsed_chunks = splitter.split_documents(raw_docs)

        # Gürültülü içerikleri ve kaynakça bölümlerini eleme
        cleaned_chunks = []
        for item in parsed_chunks:
            txt = item.page_content.strip()
            if "https://doi.org" in txt and "Dergisi" in txt:
                continue
            if txt.startswith("References") or txt.startswith("KAYNAKÇA"):
                continue
            cleaned_chunks.append(txt)

        print(f"[+] Toplam {len(cleaned_chunks)} işlenebilir parça üretildi.")

        print(f"[+] Embedding modeli aktif ediliyor: {EMBED_MODEL_IDENTIFIER}")
        self.embedder = SentenceTransformer(EMBED_MODEL_IDENTIFIER)

        print(f"[+] Veritabanı tabloları güncelleniyor: {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS documents")
            cursor.execute(
                """
                CREATE TABLE documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    embedding TEXT
                )
            """
            )

            for chunk in cleaned_chunks:
                encoded_vec = self.embedder.encode(
                    chunk, normalize_embeddings=True
                ).tolist()
                cursor.execute(
                    "INSERT INTO documents (content, embedding) VALUES (?, ?)",
                    (chunk, json.dumps(encoded_vec)),
                )
            conn.commit()

        print("[+] Vektör indeksleme işlemi tamamlandı.\n")

    def query_vector_store(
        self, query_text: str, k: int = RETRIEVAL_LIMIT
    ) -> List[Tuple[str, float]]:
        """Sorguyu vektöre çevirir ve skaler çarpım ile en yakın sonuçları getirir."""
        clean_input = query_text.strip().strip('"').strip("'")
        target_vec = self.embedder.encode(
            clean_input, normalize_embeddings=True
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content, embedding FROM documents")
            dataset = cursor.fetchall()

        matched_records = []
        for text_content, vec_json in dataset:
            stored_vec = np.array(json.loads(vec_json))
            similarity = np.dot(target_vec, stored_vec)
            matched_records.append((text_content, similarity))

        matched_records.sort(key=lambda item: item[1], reverse=True)
        return matched_records[:k]

    def init_llm_engine(self):
        """Foundry Local Runtime ve ilgili LLM modelini yükler."""
        print(f"[+] Microsoft Foundry Local ve LLM ({TARGET_LLM}) başlatılıyor...")
        FoundryLocalManager.initialize(
            Configuration(app_name="local-rag-app")
        )
        llm_instance = FoundryLocalManager.instance.catalog.get_model(
            TARGET_LLM
        )
        llm_instance.download()
        llm_instance.load()
        self.chat_client = llm_instance.get_chat_client()

    def run_interactive_loop(self):
        """Kullanıcı sorgularını kabul eden ana döngü."""
        print("\n" + "#" * 60)
        print(" LOCAL RAG SYSTEM READY ")
        print(" Fully Offline | Local Execution | Zero Hallucination Strategy")
        print("#" * 60 + "\n")

        while True:
            try:
                user_input = input("\nSoru Sor (Çıkmak için 'q'): ")
                if user_input.lower() in ["exit", "q", "çıkış"]:
                    print("Sistem kapatılıyor...")
                    break

                if not user_input.strip():
                    continue

                # A) RETRIEVAL PHASE
                start_retrieval = time.time()
                top_matches = self.query_vector_store(user_input)
                retrieval_duration = time.time() - start_retrieval

                # Context Yapılandırması
                context_payload = "\n\n".join(
                    [
                        f"[Document Part {idx + 1}]: {data[0]}"
                        for idx, data in enumerate(top_matches)
                    ]
                )

                # B) GUARDRAIL PROMPT SYSTEM
                system_instruction = (
                    "You are a strict academic Q&A assistant. "
                    "Answer the question using ONLY the facts explicitly provided in the Document Parts. "
                    "If the answer is found, provide a concise answer AND explicitly cite which Document Part you used (e.g., [Document Part 1]). "
                    "If the answer is NOT present in the text, respond exactly: "
                    "'The provided document context does not contain information to answer this question.'"
                )

                prompt_content = f"Document Parts:\n{context_payload}\n\nQuestion: {user_input}\nAnswer:"

                # C) GENERATION PHASE
                start_gen = time.time()
                response_obj = self.chat_client.complete_chat(
                    [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt_content},
                    ]
                )
                gen_duration = time.time() - start_gen

                # D) DISPLAY OUTPUT
                print("\n>>> CEVAP <<<")
                print(response_obj.choices[0].message.content)
                print("=" * 50)
                print(
                    f"PERFORMANS: Arama: {retrieval_duration:.3f}s | "
                    f"Üretim: {gen_duration:.2f}s | "
                    f"Toplam: {retrieval_duration + gen_duration:.2f}s"
                )
                print("=" * 50)

            except Exception as err:
                print(f"[!] Çalışma zamanı hatası: {err}")


if __name__ == "__main__":
    app = LocalRAGAssistant(pdf_path=DOCUMENT_FILE, db_path=DATABASE_FILE)
    app.prepare_vector_store()
    app.init_llm_engine()
    app.run_interactive_loop()