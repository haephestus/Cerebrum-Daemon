"""
test_archive_path.py
--------------------
Inspects a cerebrum .archives directory — the root itself IS the Chroma
store, not a parent of domain/subject subfolders.

Default path:
    ~/.local/share/cerebrum/study_bubbles/genetics/notes/.archives

Run with:
    python test_archive_path.py
    python test_archive_path.py --root /path/to/.archives
    python test_archive_path.py --limit 5
    python test_archive_path.py --files
"""

import argparse
import json
import sys
from pathlib import Path

ARCHIVES_ROOT = (
    Path.home() / ".local/share/cerebrum/study_bubbles/genetics/notes/.archives"
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _fmt_bytes(n: int) -> str:
    size: float = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ── 1. Path existence check ──────────────────────────────────────────────────


def test_root_exists(root: Path) -> bool:
    print("\n" + "=" * 60)
    print("TEST: Archive root exists")
    print("=" * 60)
    print(f"  Path : {root}")
    if root.exists() and root.is_dir():
        total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        print(f"  ✓  Directory found  ({_fmt_bytes(total)} on disk)")
        return True
    print("  ✗  NOT FOUND. Check the path.")
    return False


# ── 2. Chroma DB presence check ──────────────────────────────────────────────


def test_chroma_db(root: Path) -> bool:
    print("\n" + "=" * 60)
    print("TEST: Chroma SQLite database")
    print("=" * 60)
    sqlite = root / "chroma.sqlite3"
    if sqlite.exists():
        print(f"  ✓  chroma.sqlite3 found  ({_fmt_bytes(sqlite.stat().st_size)})")
        return True
    print("  ✗  chroma.sqlite3 NOT found — store may be uninitialised or corrupt.")
    return False


# ── 3. Collection contents (no embeddings needed) ────────────────────────────


def _read_chroma_raw(root: Path, limit: int) -> dict:
    try:
        import chromadb  # noqa: PLC0415
    except ImportError:
        return {"error": "chromadb not installed — run: pip install chromadb"}
    try:
        client = chromadb.PersistentClient(path=str(root))
        result: dict = {}
        for col in client.list_collections():
            raw = col.get(limit=limit, include=["documents", "metadatas"])
            result[col.name] = {
                "total": col.count(),
                "sample_docs": raw.get("documents") or [],
                "sample_metas": raw.get("metadatas") or [],
            }
        return result
    except Exception as exc:
        return {"error": str(exc)}


def test_chroma_collections(root: Path, limit: int = 3) -> None:
    print("\n" + "=" * 60)
    print(f"TEST: Chroma collections  (showing up to {limit} docs each)")
    print("=" * 60)

    data = _read_chroma_raw(root, limit)

    if "error" in data:
        print(f"  ✗  {data['error']}")
        return

    if not data:
        print("  ✗  No collections found inside this store.")
        return

    print(f"  ✓  {len(data)} collection(s) found:\n")
    for col_name, info in data.items():
        print(f"  ── Collection: '{col_name}'")
        print(f"     Total documents : {info['total']}")

        if info["total"] == 0:
            print("     (empty)\n")
            continue

        for i, (doc, meta) in enumerate(
            zip(info["sample_docs"], info["sample_metas"]), start=1
        ):
            print(f"\n     [{i}] Meta    : {json.dumps(meta, ensure_ascii=False)}")
            preview = doc[:400] + ("…" if len(doc) > 400 else "")
            print(f"          Content : {preview}")
        print()


# ── 4. Raw file listing (optional) ───────────────────────────────────────────


def test_raw_file_listing(root: Path) -> None:
    print("\n" + "=" * 60)
    print("TEST: Raw file listing")
    print("=" * 60)
    files = sorted(f for f in root.rglob("*") if f.is_file())
    print(f"  {len(files)} file(s) in {root}\n")
    for f in files:
        print(f"  {f.relative_to(root)}  ({_fmt_bytes(f.stat().st_size)})")


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a cerebrum .archives store")
    parser.add_argument(
        "--root",
        type=Path,
        default=ARCHIVES_ROOT,
        help="Path to the .archives directory (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Sample documents to show per collection (default: 3)",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="Also print a raw file listing of the store",
    )
    args = parser.parse_args()

    if not test_root_exists(args.root):
        sys.exit(1)

    test_chroma_db(args.root)
    test_chroma_collections(args.root, limit=args.limit)

    if args.files:
        test_raw_file_listing(args.root)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
