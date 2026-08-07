import logging
import sys
from pathlib import Path

# Fix path indexing for local imports
sys.path.append(str(Path(__file__).resolve().parent))

# Import your core workspace tools
from common.file_util_inator import CerebrumPaths

# Adjust this import to point directly to your NoteChunkerInator location
from notes.note_util_inator import NoteChunkerInator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_live_genetics_pipeline():
    print("=" * 80)
    print("🔬 RUNNING PIPELINE VERIFICATION ON LIVE GENETICS NOTES")
    print("=" * 80)

    # 1. Target your actual home directory layout
    genetics_dir = Path(
        "/home/harbinger/.local/share/cerebrum/study_bubbles/genetics/.derived/chunked_notes"
    )

    if not genetics_dir.exists():
        print(f"❌ ERROR: Could not find directory layout at:\n   {genetics_dir}")
        return

    notes = list(genetics_dir.glob("*.md"))
    if not notes:
        print(f"❌ ERROR: No .md files found inside {genetics_dir}")
        return

    # Pick the first actual note found on your system
    target_note = notes[0]
    print(f"📖 Target file selected: {target_note.name}")
    raw_text = target_note.read_text(encoding="utf-8")

    # 2. Run through the workspace chunker pipeline
    chunker = NoteChunkerInator(generate_artifacts=False)
    _, documents = chunker.chunk(
        flattened_note=raw_text, note_id=target_note.stem, bubble_id="genetics"
    )

    # 3. Print out validation telemetry logs
    print(
        f"\n📦 Extracted {len(documents)} LangChain documents from processing framework."
    )
    sample_size = min(2, len(documents))

    print(f"Printing details for the first {sample_size} document objects:\n")
    for i in range(sample_size):
        doc = documents[i]
        print(f"▶️ DOCUMENT OBJECT index={i}")
        print(f"   🔒 Metadata Properties Dict: {doc.metadata}")
        print("   💬 Text Body Content String:")
        print("   " + "-" * 60)
        for line in doc.page_content.splitlines()[:6]:  # Show first 6 lines maximum
            print(f"   | {line}")
        if len(doc.page_content.splitlines()) > 6:
            print("   | ... [truncated text payload] ...")
        print("   " + "-" * 60)

        # Automated Sanity Check Rules
        has_byte_start = "byte_start" in doc.metadata
        has_corrupted_text = (
            "byte_start:" in doc.page_content or "chunk_index:" in doc.page_content
        )

        if has_byte_start:
            print(
                "   ✅ METADATA TRACKING: SUCCESS (byte coordinates map to metadata object dict)"
            )
        else:
            print(
                "   ❌ METADATA TRACKING: FAILED (byte coordinates are missing from object dict)"
            )

        if not has_corrupted_text:
            print(
                "   ✅ TEXT SANITIZATION: SUCCESS (raw text payload is pristine and unpolluted)"
            )
        else:
            print(
                "   🚨 TEXT SANITIZATION: FAILED (technical metadata markers leaked into text string)"
            )
        print()


if __name__ == "__main__":
    test_live_genetics_pipeline()
