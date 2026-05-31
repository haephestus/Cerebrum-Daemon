"""
test_vectorstore_langchain.py
------------------------------
Inspect a cerebrum .archives Chroma vector store using LangChain.
Displays document contents, metadata, and IDs for every collection.

Default path:
    ~/.local/share/cerebrum/study_bubbles/genetics/notes/.archives

Usage:
    python test_vectorstore_langchain.py
    python test_vectorstore_langchain.py --root /path/to/.archives
    python test_vectorstore_langchain.py --limit 10
    python test_vectorstore_langchain.py --collection my_collection
    python test_vectorstore_langchain.py --all          # fetch ALL records (no limit)
"""

import argparse
import json
import sys
from pathlib import Path

ARCHIVES_ROOT = (
    Path.home() / ".local/share/cerebrum/study_bubbles/genetics/notes/.archives"
)

SEP = "─" * 70


# ── helpers ───────────────────────────────────────────────────────────────────


def _fmt_bytes(n: int) -> str:
    size: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _print_header(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


# ── 1. Sanity checks ──────────────────────────────────────────────────────────


def check_root(root: Path) -> bool:
    _print_header("CHECK: Archive root")
    if not root.exists() or not root.is_dir():
        print(f"  ✗  NOT FOUND: {root}")
        return False
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    print(f"  ✓  {root}")
    print(f"  ✓  Total size on disk: {_fmt_bytes(total)}")
    return True


def check_sqlite(root: Path) -> bool:
    _print_header("CHECK: Chroma SQLite database")
    sqlite = root / "chroma.sqlite3"
    if sqlite.exists():
        print(f"  ✓  chroma.sqlite3 ({_fmt_bytes(sqlite.stat().st_size)})")
        return True
    print("  ✗  chroma.sqlite3 NOT found — store may be uninitialised.")
    return False


# ── 2. LangChain retrieval ────────────────────────────────────────────────────


def load_vectorstore(root: Path, collection_name: str):
    """
    Load a Chroma collection via LangChain using a dummy embeddings object.
    We skip real embeddings because we only want stored text + metadata.
    """
    from langchain_chroma import Chroma
    from langchain_core.embeddings import Embeddings

    class _NullEmbeddings(Embeddings):
        """Placeholder — we never call embed_* in read-only inspection mode."""

        def embed_documents(self, texts):
            return [[0.0]] * len(texts)

        def embed_query(self, text):
            return [0.0]

    store = Chroma(
        collection_name=collection_name,
        persist_directory=str(root),
        embedding_function=_NullEmbeddings(),
    )
    return store


def list_collections(root: Path) -> list[str]:
    """Return all collection names in the store via raw chromadb client."""
    import chromadb

    client = chromadb.PersistentClient(path=str(root))
    return [c.name for c in client.list_collections()]


def fetch_all_records(root: Path, collection_name: str, limit: int | None) -> dict:
    """
    Pull documents directly from the underlying chromadb collection.
    Returns a dict with keys: ids, documents, metadatas.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(root))
    col = client.get_collection(collection_name)
    total = col.count()

    fetch_n = total if limit is None else min(limit, total)
    raw = col.get(
        limit=fetch_n,
        include=["documents", "metadatas"],
    )
    return {
        "total": total,
        "fetched": fetch_n,
        "ids": raw.get("ids") or [],
        "documents": raw.get("documents") or [],
        "metadatas": raw.get("metadatas") or [],
    }


# ── 3. Display ────────────────────────────────────────────────────────────────


def print_records(collection_name: str, data: dict) -> None:
    _print_header(f"COLLECTION: '{collection_name}'")
    print(f"  Total records  : {data['total']}")
    print(f"  Showing        : {data['fetched']}")

    if data["total"] == 0:
        print("  (empty collection)")
        return

    for i, (doc_id, doc, meta) in enumerate(
        zip(data["ids"], data["documents"], data["metadatas"]),
        start=1,
    ):
        print(f"\n  {SEP}")
        print(f"  Record #{i}")
        print(f"  {SEP}")
        print(f"  ID       : {doc_id}")
        print(f"  Metadata : {json.dumps(meta, indent=4, ensure_ascii=False)}")
        print(f"  Content  :")
        # indent each line for readability
        for line in (doc or "").splitlines():
            print(f"    {line}")

    print(f"\n  {SEP}")


# ── 4. LangChain similarity search smoke-test ─────────────────────────────────


def smoke_test_similarity(root: Path, collection_name: str) -> None:
    """
    Verifies that LangChain's similarity_search works on the store.
    Uses a generic query — just checks that the integration is wired correctly.
    """
    _print_header(f"SMOKE TEST: LangChain similarity_search on '{collection_name}'")
    try:
        store = load_vectorstore(root, collection_name)
        # k=1: we only want to confirm retrieval works, not flood the output
        results = store.similarity_search("genetics", k=1)
        if results:
            doc = results[0]
            print("  ✓  similarity_search returned a result")
            print(f"  Content preview : {doc.page_content[:300]}")
            print(f"  Metadata        : {doc.metadata}")
        else:
            print("  ✓  similarity_search ran (no results for query 'genetics')")
    except Exception as exc:
        print(f"  ✗  similarity_search failed: {exc}")
        print(
            "     (This is expected when using NullEmbeddings — "
            "swap in your real embeddings to enable semantic search.)"
        )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a cerebrum Chroma store via LangChain"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ARCHIVES_ROOT,
        help="Path to the .archives directory (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max records to display per collection (default: 5)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch ALL records (overrides --limit)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Inspect a specific collection by name (default: all)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Also run a LangChain similarity_search smoke test",
    )
    args = parser.parse_args()

    # ── sanity checks
    if not check_root(args.root):
        sys.exit(1)
    check_sqlite(args.root)

    # ── resolve collections to inspect
    try:
        all_collections = list_collections(args.root)
    except Exception as exc:
        print(f"\n  ✗  Could not list collections: {exc}")
        sys.exit(1)

    if not all_collections:
        print("\n  ✗  No collections found in this store.")
        sys.exit(0)

    if args.collection:
        if args.collection not in all_collections:
            print(f"\n  ✗  Collection '{args.collection}' not found.")
            print(f"     Available: {all_collections}")
            sys.exit(1)
        target_collections = [args.collection]
    else:
        target_collections = all_collections

    print(f"\n  Collections in store : {all_collections}")

    # ── fetch + display records
    limit = None if args.all else args.limit

    for col_name in target_collections:
        try:
            data = fetch_all_records(args.root, col_name, limit)
            print_records(col_name, data)
        except Exception as exc:
            print(f"\n  ✗  Failed to read '{col_name}': {exc}")

    # ── optional smoke test
    if args.smoke:
        for col_name in target_collections:
            smoke_test_similarity(args.root, col_name)

    print(f"\n{'═' * 70}")
    print("  Done.")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
