"""MuseumMind - RAG core (LlamaIndex + ChromaDB + OpenAI)."""
import os
import re
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from llama_index.core import (Settings, SimpleDirectoryReader, StorageContext,
                              VectorStoreIndex)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
CHROMA_DIR = BASE / "chroma_db"
COLLECTION = "museum"

# ---- החלטות התכנון ----
CHUNK_SIZE = 500        # טוקנים (SentenceSplitter)
CHUNK_OVERLAP = 100     # חפיפה
TOP_K = 5               # כמה קטעים מוחזרים
MIN_SCORE = 0.28       # סף רלוונטיות: מתחתיו - "אין מידע", בלי לקרוא ל-LLM
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

NO_INFO = "אין לי מספיק מידע במסמכי המוזיאון כדי לענות על השאלה הזו."

QA_PROMPT = """אתה MuseumMind, העוזר החכם של "מוזיאון מסע בזמן".
ענה על השאלה אך ורק על סמך הקטעים שלהלן. אל תשתמש בידע חיצוני.
אם הקטעים אינם מכילים את התשובה, או מכילים אותה רק חלקית - אמור זאת במפורש
ואל תנחש. ענה בעברית, בקצרה ובבהירות.

קטעים:
{context}

שאלה: {question}
תשובה:"""


def _setup():
    load_dotenv(BASE / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("חסר OPENAI_API_KEY. העתיקי את .env.example ל-.env והדביקי מפתח.")
    Settings.llm = OpenAI(model=LLM_MODEL, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=EMBED_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _store():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client, ChromaVectorStore(chroma_collection=client.get_or_create_collection(COLLECTION))


def build_index():
    """מוחק את האינדקס הקיים ובונה מחדש מתיקיית data."""
    _setup()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    _, vs = _store()
    docs = SimpleDirectoryReader(str(DATA_DIR), required_exts=[".txt"], encoding="utf-8").load_data()
    return VectorStoreIndex.from_documents(
        docs, storage_context=StorageContext.from_defaults(vector_store=vs))


def load_index():
    _setup()
    _, vs = _store()
    if vs._collection.count() == 0:
        raise RuntimeError("האינדקס ריק. הריצי: python ingest.py")
    return VectorStoreIndex.from_vector_store(vs)


def expand_question(question: str) -> str:
    """הרחבת שאלה בקריאת LLM - רק לפי בקשה מפורשת, לא ברירת מחדל."""
    prompt = ("הרחב את השאלה הקצרה הבאה לשאלה חיפוש מלאה אחת בעברית על מוזיאון, "
              "בלי להוסיף עובדות חדשות. החזר רק את השאלה.\nשאלה: " + question)
    return str(Settings.llm.complete(prompt)).strip()


ID_RE = re.compile(r"(?:TM|CR|RP)-[\d-]+", re.I)


def _id_lookup(question: str):
    """כלל פשוט (בלי AI): מספר קטלוג/דוח בשאלה -> חיפוש טקסט מדויק ב-ChromaDB."""
    ids = ID_RE.findall(question)
    if not ids:
        return []
    _, vs = _store()
    res = vs._collection.get(where_document={"$contains": ids[0].upper()},
                             include=["documents", "metadatas"])
    return [{"file": m.get("file_name", "?"), "score": 1.0, "text": d}
            for d, m in zip(res["documents"], res["metadatas"])][:TOP_K]


def ask(index, question: str, expand: bool = False) -> dict:
    """Similarity search -> סינון לפי סף -> תשובה עם מקורות.
    כללים לשאלות קצרות: מספר קטלוג -> התאמה מדויקת; עד 2 מילים -> "מה זה ...?"."""
    search_q = expand_question(question) if expand else question
    if not expand and len(question.split()) <= 2 and not ID_RE.search(question):
        search_q = f"מה זה {question.strip(' ?')}?"
    retriever = index.as_retriever(similarity_top_k=TOP_K)  # ברירת מחדל = Similarity Search
    nodes = [n for n in retriever.retrieve(search_q) if (n.score or 0) >= MIN_SCORE]
    sources = _id_lookup(question) or [
        {"file": n.node.metadata.get("file_name", "?"),
         "score": round(n.score, 3), "text": n.node.get_content()} for n in nodes]
    nodes = sources
    if not nodes:
        return {"answer": NO_INFO, "sources": [], "search_query": search_q}
    context = "\n---\n".join(s["text"] for s in sources)
    answer = str(Settings.llm.complete(QA_PROMPT.format(context=context, question=question))).strip()
    return {"answer": answer, "sources": sources, "search_query": search_q}
