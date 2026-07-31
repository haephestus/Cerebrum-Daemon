import chromadb


def purge_untimestamped_duplicates(archives_path: str, note_id: str) -> int:
    """
    One-time cleanup: old archive entries inserted before archived_at
    existed have no recoverable insert time and (for notes archived
    under the pre-fingerprint-id scheme) may be exact-content duplicates
    of each other. Keeps one copy per unique page_content, deletes the
    rest. Returns the number of ids deleted.
    """
    client = chromadb.PersistentClient(path=archives_path)
    try:
        collection = client.get_collection(name=note_id)
    except Exception:
        return 0

    raw = collection.get()
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []

    seen_content: set[str] = set()
    to_delete: list[str] = []

    for doc_id, content in zip(ids, documents):
        if content in seen_content:
            to_delete.append(doc_id)
        else:
            seen_content.add(content)

    if to_delete:
        collection.delete(ids=to_delete)

    return len(to_delete)
