# test_chroma_viewer.py
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

ARCHIVES_ROOT = Path.home() / ".local/share/cerebrum/knowledgebase/archives"
EMBEDDING_MODEL = "qwen3-embedding:4b-q4_K_M"


def get_all_collections(archives_root: Path) -> list[dict]:
    """Walk archives and find all chroma collections (domain/subject folders)."""
    collections = []
    for subject_dir in archives_root.glob("*/*"):
        if subject_dir.is_dir():
            domain = subject_dir.parent.name
            subject = subject_dir.name
            collections.append(
                {
                    "domain": domain,
                    "subject": subject,
                    "path": subject_dir,
                }
            )
    return collections


def peek_collection(
    domain: str, subject: str, persist_dir: Path, embedding_model: str, k: int = 3
):
    """Load a Chroma collection and print its documents."""
    print(f"\n{'='*60}")
    print(f"  DOMAIN  : {domain}")
    print(f"  SUBJECT : {subject}")
    print(f"  PATH    : {persist_dir}")
    print(f"{'='*60}")

    try:
        # List all collection names in this directory
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
        collection_names = [c.name for c in client.list_collections()]
        print(f"  Collections found: {collection_names}\n")

        for col_name in collection_names:
            print(f"  --- Collection: '{col_name}' ---")
            db = Chroma(
                embedding_function=OllamaEmbeddings(model=embedding_model),
                collection_name=col_name,
                persist_directory=str(persist_dir),
            )
            # Get total count
            total = db._collection.count()
            print(f"  Total documents: {total}")

            if total == 0:
                print("  (empty collection)")
                continue

            # Peek at first k documents without a query
            raw = db._collection.get(limit=k, include=["documents", "metadatas"])

            docs = raw["documents"] or []
            metas = raw["metadatas"] or []

            for i, (doc, meta) in enumerate(zip(docs, metas)):
                print(f"\n  [{i+1}] Metadata : {meta}")
                print(f"       Content  : {doc[:300]}{'...' if len(doc) > 300 else ''}")

    except Exception as e:
        print(f"  ERROR loading collection: {e}")


def main():
    print(f"\nScanning archives at: {ARCHIVES_ROOT}\n")

    if not ARCHIVES_ROOT.exists():
        print(f"ERROR: Archives root not found at {ARCHIVES_ROOT}")
        return

    collections = get_all_collections(ARCHIVES_ROOT)

    if not collections:
        print("No domain/subject folders found.")
        return

    print(f"Found {len(collections)} collection(s):\n")
    for c in collections:
        print(f"  {c['domain']}/{c['subject']}")

    # Peek into each one
    for c in collections:
        peek_collection(
            domain=c["domain"],
            subject=c["subject"],
            persist_dir=c["path"],
            embedding_model=EMBEDDING_MODEL,
            k=3,  # number of docs to preview per collection
        )

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
