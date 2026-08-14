import re
import shutil
from pathlib import Path

from platformdirs import PlatformDirs

"""
file_util_inator.py

Purpose: 
    Exposes file paths, and handles all file related manipulations
    regarding what is available in the knowledgebase.
"""


class CerebrumPaths:
    """
    Exposes necessary file paths and makes it easier to define
    config level control concerning default file locations.
    """

    def __init__(self, app_name: str = "cerebrum"):
        dirs = PlatformDirs(app_name)
        self.DATA_ROOT = Path(dirs.user_data_dir)
        self.CONFIG_ROOT = Path(dirs.user_config_dir)
        self.CACHE_ROOT = Path(dirs.user_cache_dir)
        # Cerebrum paths
        self.KB_ROOT = self.DATA_ROOT / "knowledgebase"
        self.BUBBLES_ROOT = self.DATA_ROOT / "study_bubbles"
        self.LOGS_ROOT = self.DATA_ROOT / "logs"

    def init_cerebrum_dirs(self):
        """Ensure all top-level directories exist."""
        for d in [
            self.DATA_ROOT,
            self.KB_ROOT,
            self.BUBBLES_ROOT,
            self.LOGS_ROOT,
        ]:
            d.mkdir(exist_ok=True)

    # ------------- BUBBLE PATHS -------------------------------------------

    def init_bubble_dirs(self, bubble_id: str):
        """Create all directories for a new study bubble."""
        bubble_dir = self.bubble_path(
            bubble_id
        )  # fixed: was / bubble_id (double-nested)
        for d in [
            bubble_dir / "chat",
            bubble_dir / "notes",
            bubble_dir / "engrams",  # fixed: was "assesments"
            bubble_dir / ".derived",
        ]:
            d.mkdir(parents=True, exist_ok=True)
            (d / ".archives").mkdir(parents=True, exist_ok=True)

    def bubbles_root_dir(self) -> Path:
        """Return bubbles root directory."""
        return self.BUBBLES_ROOT

    def bubble_path(self, bubble_id: str) -> Path:
        """Return the path of a single bubble."""
        return self.BUBBLES_ROOT / bubble_id

    def note_root_dir(self, bubble_id: str) -> Path:
        """Return notes root directory."""
        return self.bubble_path(bubble_id) / "notes"

    def note_path(self, bubble_id: str, filename: str) -> Path:
        """Return path to a single note."""
        return self.bubble_path(bubble_id) / "notes" / filename

    def note_archive_path(self, bubble_id: str) -> Path:
        """Return bubble-specific note archives directory."""
        return self.bubble_path(bubble_id) / "notes" / ".archives"

    def chat_root_dir(self, bubble_id: str) -> Path:
        """Return bubble-specific chats directory."""
        return self.bubble_path(bubble_id) / "chat"

    def chat_archives_path(self, bubble_id: str) -> Path:
        """Return bubble-specific chat archives directory."""
        return self.bubble_path(bubble_id) / "chat" / ".archives"

    def engram_path(self, bubble_id: str) -> Path:
        """Return bubble-specific engrams directory."""
        return self.bubble_path(bubble_id) / "engrams"

    def engram_archives_path(self, bubble_id: str) -> Path:
        """Return bubble-specific engrams archives directory."""
        return self.bubble_path(bubble_id) / "engrams" / ".archives"

    # ------------- .DERIVED PATHS (computed artefacts) --------------------

    def derived_root(self, bubble_id: str) -> Path:
        """Hidden dir for all computed artefacts inside a bubble."""
        return self.bubble_path(bubble_id) / ".derived"

    def chunked_note_path(self, bubble_id: str, note_id: str) -> Path:
        """Chunk file for a single note."""
        return self.derived_root(bubble_id) / "chunked_notes" / note_id

    def chunked_note_file(self, bubble_id: str, note_id: str) -> Path:
        """Chunk file for a single note."""
        return (
            self.derived_root(bubble_id) / "chunked_notes" / note_id / f"{note_id}.md"
        )

    def note_analysis_dir(self, bubble_id: str, note_id: str) -> Path:
        """Dir holding per-chunk analysis results for a note."""
        # Strips any rogue .json extension if filename leaks into the note_id argument
        clean_note_id = str(note_id).removesuffix(".json")
        return self.derived_root(bubble_id) / "analyses" / clean_note_id

    def invalidate_note_derived(self, bubble_id: str, note_id: str) -> None:
        """Remove all derived data for a note. Call this when a note's content changes."""
        for p in [
            self.chunked_note_path(bubble_id, note_id),
            self.note_analysis_dir(bubble_id, note_id),
        ]:
            if p.exists():
                shutil.rmtree(p)

    # ------------- KNOWLEDGEBASE PATHS ------------------------------------

    def kb_root_dir(self) -> Path:
        """Return knowledgebase root directory."""
        return self.DATA_ROOT / "knowledgebase"

    def kb_source_files_path(self) -> Path:
        return self.kb_root_dir() / "source_files"

    def kb_artifacts_path(
        self,
        domain,
        subject,
        sanitised_name,
    ) -> Path:
        return (
            self.kb_root_dir()
            / "markdown_artifacts"
            / domain
            / subject
            / sanitised_name
        )

    def kb_archives_path(self) -> Path:
        return self.kb_root_dir() / "archives"

    # ------------- LOGS & CONFIG ------------------------------------------

    def logs_root_dir(self) -> Path:
        return self.DATA_ROOT / "logs"

    def config_root_dir(self) -> Path:
        return self.CONFIG_ROOT

    # ------------- CACHE --------------------------------------------------

    def cache_root_dir(self) -> Path:
        return self.CACHE_ROOT


def file_walker_inator(root: Path, max_depth: int = 4):
    """
    walks the  knowledgebase root directory, in order to give context
    of the available domains to the llm, so it can determinstically
    classify documents
    """

    def recurse_inator(path: Path, parts: list[str]):
        for file in path.glob("*"):
            if file.is_file():
                yield {
                    "domain": parts[0] if len(parts) > 0 else None,
                    "subject": parts[1] if len(parts) > 1 else None,
                    "topic": parts[2] if len(parts) > 2 else None,
                    "subtopic": parts[3] if len(parts) > 3 else None,
                    "filepath": file,
                    "filename": file.name,
                    "filestem": file.stem,
                    "file-ext": file.suffix,
                }
            elif file.is_dir() and len(parts) < max_depth:
                yield from recurse_inator(file, parts + [file.name])

    yield from recurse_inator(root, [])


UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


# TODO: potential target for deprecation (file_registry does same job)
def knowledgebase_index_inator(root: Path):
    domains, subjects, topics, subtopics = set(), set(), set(), set()
    available_files = []

    for info in file_walker_inator(root):
        # skip if any part is a UUID
        skip = False
        for part in [info["domain"], info["subject"], info["topic"], info["subtopic"]]:
            if part and UUID_PATTERN.fullmatch(part):
                skip = True
                break
        if skip:
            continue

        available_files.append(info["filestem"])
        if info["domain"]:
            domains.add(info["domain"])
        if info["subject"]:
            subjects.add(info["subject"])
        if info["topic"]:
            topics.add(info["topic"])
        if info["subtopic"]:
            subtopics.add(info["subtopic"])

    return {
        "domains": sorted(domains),
        "subjects": sorted(subjects),
        "topics": sorted(topics),
        "subtopics": sorted(subtopics),
    }, available_files
