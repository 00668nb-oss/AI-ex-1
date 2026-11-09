from rag import build_index, CHROMA_DIR

if __name__ == "__main__":
    build_index()
    print(f"האינדקס נבנה ונשמר ב-{CHROMA_DIR}")
