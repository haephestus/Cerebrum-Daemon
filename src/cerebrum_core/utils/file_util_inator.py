import re
from pathlib import Path

from platformdirs import PlatformDirs

"""
file_util_inator.py

Purpose: 
    Exposes file paths, and handles all file related manipulations
    regarding what is available in the knowledgebase.
"""


# init dirs for server
class CerebrumPaths:
    """
    Exposes necesary file paths and makes it easier to define
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
        """Ensure all top-level directories exist"""
        for d in [
            self.DATA_ROOT,
            self.KB_ROOT,
            self.BUBBLES_ROOT,
            self.LOGS_ROOT,
        ]:
            d.mkdir(exist_ok=True)

    # ------------- HANDLE BUBBLES PATHS ---------------------------
    def init_bubble_dirs(self, bubble_id):
        """
        Handles creation of bubble specific folders when a new study bubble
        is created
        """
        bubble_dir = self.bubble_path(bubble_id) / bubble_id

        # Create sub-dirs
        chat_dir = bubble_dir / "chat"
        notes_dir = bubble_dir / "notes"
        assesments_dir = bubble_dir / "assesments"

        for d in [chat_dir, notes_dir, assesments_dir]:
            d.mkdir(parents=True, exist_ok=True)
            (d / ".archives").mkdir(parents=True, exist_ok=True)

    def bubbles_root_dir(self) -> Path:
        """Return bubbles root directory"""
        return self.BUBBLES_ROOT

    def bubble_path(self, bubble_id):
        """Return the path of a single bubble"""
        BUBBLE_PATH = self.BUBBLES_ROOT / bubble_id
        return BUBBLE_PATH

    def note_root_dir(self, bubble_id):
        """Return notes root directory"""
        return self.bubble_path(bubble_id) / "notes"

    def note_path(self, bubble_id: str, filename: str):
        """Return path to a sinlge note"""
        return self.bubble_path(bubble_id) / "notes" / filename

    def note_archive_path(self, bubble_id):
        """Return bubble specific note archives directory"""
        return self.bubble_path(bubble_id) / "notes" / ".archives"

    def chat_root_dir(self, bubble_id):
        """Return  bubble specific chats directory"""
        return self.bubble_path(bubble_id) / "chat"

    def chat_archives_path(self, bubble_id):
        """Return  bubble specific chat archives directory"""
        return self.bubble_path(bubble_id) / "chat" / ".archives"

    def assesment_path(self, bubble_id):
        """Returns bubble specific assesment directory"""
        return self.bubble_path(bubble_id) / "assesments"

    def assesment_archives_path(self, bubble_id):
        """Returns bubble specific assesment archives directory"""
        return self.bubble_path(bubble_id) / "assesments" / ".archives"

    # ------------- HANDLE KNOWLEDGEBASE PATHS ---------------------------
    def kb_root_dir(self) -> Path:
        KB_DIR = self.DATA_ROOT / "knowledgebase"
        return KB_DIR

    def kb_source_files_path(self) -> Path:
        return self.kb_root_dir() / "source_files"

    def kb_artifacts_path(self) -> Path:
        return self.kb_root_dir() / "markdown_artifacts"

    def kb_archives_path(self):
        return self.kb_root_dir() / "archives"

    def logs_root_dir(self) -> Path:
        LOGS_DIR = self.DATA_ROOT / "logs"
        return LOGS_DIR

    def config_root_dir(self) -> Path:
        return self.CONFIG_ROOT

    # ------------- HANDLE CACHE PATHS ---------------------------
    def cache_root_dir(self) -> Path:
        return self.CACHE_ROOT

    def analysis_cache_path(self, bubble_id) -> Path:
        CACHE_DIR = self.CACHE_ROOT / "analysis_cache" / bubble_id
        return CACHE_DIR

    def note_cache_path(self, bubble_id) -> Path:
        return self.cache_root_dir() / bubble_id


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
