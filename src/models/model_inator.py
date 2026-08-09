import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class Subtopic(BaseModel):
    name: str
    description: str


class Topic(BaseModel):
    name: str
    subtopics: List[Subtopic] = []
    description: Optional[str] = None

    def add_subtopic(self, subtopic_name: str, description: str):
        subtopic = Subtopic(name=subtopic_name, description=description)
        self.subtopics.append(subtopic)
        return subtopic


class Subject(BaseModel):
    name: str
    description: str
    topics: List[Topic] = []

    def add_topic(self, topic_name: str, description: str):
        topic = Topic(name=topic_name, description=description)
        self.topics.append(topic)
        return topic


class Domain(BaseModel):
    name: str
    description: str
    subjects: List[Subject] = []

    def add_subject(self, subject_name: str, description: str):
        subject = Subject(name=subject_name, description=description)
        self.subjects.append(subject)
        return subject


class KnowledgeBase(BaseModel):
    name: str
    description: str
    domains: List[Domain] = []

    def add_domain(self, domain_name: str, description: str):
        domain = Domain(name=domain_name, description=description)
        self.domains.append(domain)
        return domain


#############################################################################
#                                                                           #
#                        USER CONFIG MODELS                                 #
#                                                                           #
#############################################################################


class User(BaseModel):
    name: str
    password: str
    selected_chat_model: str = ""
    selected_embedding_model: str = ""


class ModelConfig(BaseModel):
    chat_model: Optional[str] = None
    embedding_model: Optional[str] = None
    cloud_model: Optional[str] = None


class OllamaConfig(BaseModel):
    url: str = "https://ollama.com/download"
    api_key: str = ""
    toggle_cloud: bool = False


class UserConfig(BaseModel):
    models: ModelConfig = Field(default_factory=ModelConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


#############################################################################
#                                                                           #
#                        MODELS NEEDED FOR RAG                              #
#                                                                           #
#############################################################################


class Subquery(BaseModel):
    text: str
    domain: Optional[str] = None
    subject: Optional[str] = None


class TranslatedQuery(BaseModel):
    rewritten: str
    domain: Optional[str | List[str]] = None
    subject: Optional[str | List[str]] = None
    subqueries: List[Subquery]


# Document types the ingestion router dispatches on. `unknown` is the safe
# fallback when the classifier is unsure or emits something off-vocabulary —
# downstream type-specific parsers treat it as "use the generic path".
DOC_TYPES = {
    "textbook",
    "exam_paper",
    "scientific_article",
    "notes",
    "reference",
    "unknown",
}

# Synonyms an LLM tends to emit, mapped onto the controlled vocabulary.
_DOC_TYPE_ALIASES = {
    "exam": "exam_paper",
    "past_paper": "exam_paper",
    "test": "exam_paper",
    "paper": "scientific_article",
    "article": "scientific_article",
    "journal": "scientific_article",
    "journal_article": "scientific_article",
    "research_paper": "scientific_article",
    "book": "textbook",
    "note": "notes",
    "manual": "reference",
    "handbook": "reference",
    "dictionary": "reference",
}


class FileMetadata(BaseModel):
    title: str
    domain: str
    subject: str
    authors: str | List[str]
    keywords: str | List[str]
    doc_type: str = "unknown"

    @field_validator("doc_type", mode="before")
    @classmethod
    def _normalise_doc_type(cls, v):
        if not v:
            return "unknown"
        normalised = str(v).strip().lower().replace("-", "_").replace(" ", "_")
        normalised = _DOC_TYPE_ALIASES.get(normalised, normalised)
        return normalised if normalised in DOC_TYPES else "unknown"


class Chunk(BaseModel):
    pass


#############################################################################
#                                                                           #
#                          MODELS FOR NOTES                                 #
#                                                                           #
#############################################################################


class ContentDiff(BaseModel):
    version: float
    ts: datetime
    ops: List[Dict[str, Any]]


class InkDiffOp(str, Enum):
    ADD = "add"
    REMOVE = "remove"
    MODIFY = "modify"


class InkDiff(BaseModel):
    version: float
    ts: datetime
    ops: List[Dict[str, Any]]


class PageHistory(BaseModel):
    """A page's diff journals — the on-disk `history.json` sidecar."""

    content: List[ContentDiff] = Field(default_factory=list)
    ink: List[InkDiff] = Field(default_factory=list)


class PageManifest(BaseModel):
    """The page folder's `manifest.json`: identity + per-page versioning + sync
    clocks. A page is the unit of sync/merge (gap 1): `version_vector` is the
    page's per-replica logical clocks; ink is additive (stroke-id union) so it
    rarely conflicts."""

    page_id: str = ""
    page_index: int = 0
    content_hash: str = ""
    content_version: float = 0
    ink_hash: str = ""
    ink_version: float = 0
    version_vector: Dict[str, int] = Field(default_factory=dict)
    last_modified: datetime = Field(default_factory=lambda: datetime.now())


# The in-memory view of a page FOLDER (analysis, content, history, ink, manifest).
# Kept flat (`document` dict, `ink` list) so the merge engine and chunker address
# it directly; `analysis.json` stays a pure sidecar (write/read_page_analysis) and
# is NOT loaded here — it's large and only the analysis routes need it.
class Page(BaseModel):
    """One page of a note: an AppFlowy document subtree + its ink + history +
    manifest. Pages are the hard boundary above chunks (page > chunk > block),
    the unit of per-page analysis, and the unit of offline-sync merge."""

    # Stable identity assigned by the front end and stored in the page manifest;
    # it is ALSO the page's folder name (folders are never renamed — order lives
    # in the note manifest's page_order map, so reorder/delete don't touch it).
    page_id: str
    # Display order. Mirrored into the note manifest's page_order hashtable.
    page_index: int = 0
    document: Dict[str, Any] = Field(default_factory=dict)
    ink: List[Dict[str, Any]] = Field(default_factory=list)
    history: PageHistory = Field(default_factory=PageHistory)
    metadata: PageManifest = Field(default_factory=PageManifest)


class NoteManifest(BaseModel):
    """The note folder's top-level `manifest.json`: note identity, the
    note-level analysis overview + note-level version (the overview cache keys
    off `content_version`), the sync clock, and `page_order` — a page_id→index
    hashtable that carries display order WITHOUT encoding it in folder names."""

    title: str = ""
    note_id: str = ""
    bubble_id: str = ""
    analyse_note: bool = True
    content_hash: str = ""
    content_version: float = 0
    ink_hash: str = ""
    ink_version: float = 0
    # Sync-ready (gap 1): per-replica logical clocks {replica_id: counter}. A
    # writer bumps only its own slot; the merge engine compares vectors to tell
    # "newer" from "concurrent".
    version_vector: Dict[str, int] = Field(default_factory=dict)
    last_modified: datetime = Field(default_factory=lambda: datetime.now())
    # Note-level analysis overview (set by write_note_overview); preserved across
    # plain saves since it's not part of what a writer sends.
    overview: Optional[Dict[str, Any]] = None
    # page_id -> display index. The single source of truth for page order.
    page_order: Dict[str, int] = Field(default_factory=dict)


# WARN: this is the note storage model — pages are disk-truth, everything else
# derives from them. `manifest` mirrors the note folder's manifest.json.
class Note(BaseModel):
    manifest: NoteManifest = Field(default_factory=NoteManifest)
    pages: List[Page] = Field(default_factory=list)

    # Note-level scalars surfaced at the TOP LEVEL of the serialized note (not
    # just under `manifest`) — `computed_field` so they appear in responses. This
    # keeps clients that read `note.title` / `note.version` working while
    # `manifest`/`pages` carry the structured truth. In-app these also keep the
    # many `note.note_id` / `note.title` reads as one-liners; writes go through
    # `note.manifest.*`.
    @computed_field
    @property
    def note_id(self) -> str:
        return self.manifest.note_id

    @computed_field
    @property
    def title(self) -> str:
        return self.manifest.title

    @computed_field
    @property
    def bubble_id(self) -> str:
        return self.manifest.bubble_id

    @computed_field
    @property
    def analyse_note(self) -> bool:
        return self.manifest.analyse_note

    @computed_field
    @property
    def version(self) -> float:
        return self.manifest.content_version

    @computed_field
    @property
    def content(self) -> Dict[str, Any]:
        """Compat mirror of the FIRST page as the old `{"document": {...}}`
        wrapper, so a client still reading `note.content.document` keeps working
        while it migrates to `pages`. Always a valid document (with `children`)
        so a new/empty page can't surface a null the client `!`-asserts on."""
        ordered = sorted(self.pages, key=lambda p: p.page_index)
        doc = ordered[0].document if ordered else {}
        if not (isinstance(doc, dict) and isinstance(doc.get("children"), list)):
            doc = {
                "type": "page",
                "children": [
                    {"type": "paragraph", "data": {"delta": [{"insert": ""}]}}
                ],
            }
        return {"document": doc}

    @computed_field
    @property
    def ink(self) -> List[Dict[str, Any]]:
        """Compat mirror of the first page's ink (never null)."""
        ordered = sorted(self.pages, key=lambda p: p.page_index)
        return ordered[0].ink if ordered else []


class NoteInput(BaseModel):
    """What a client POSTs on create/update: the whole note as pages. The
    client holds all pages loaded, so it sends them all — the server reconciles
    (edit/add/delete/reorder) by page_id. Server-managed fields on each Page
    (metadata/history) are ignored/overwritten; only page_id, page_index,
    document and ink are read from the input.

    Back-compat: a client that hasn't migrated its SEND path yet may still POST
    the old flat body `{title, content: {document}, ink}` with no `pages`. The
    validator below turns that into a single page, so an un-migrated client
    can't accidentally submit an empty `pages` (which on update would wipe the
    note)."""

    title: str
    note_id: str = ""
    bubble_id: str = ""
    pages: List[Page] = Field(default_factory=list)
    # Legacy flat body (ignored when `pages` is provided).
    content: Optional[Dict[str, Any]] = None
    ink: List[Dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _synth_pages_from_legacy(self):
        if not self.pages and self.content is not None:
            doc = self.content.get("document", {}) if isinstance(self.content, dict) else {}
            self.pages = [
                Page(page_id="p1", page_index=0, document=doc, ink=self.ink or [])
            ]
        return self


class NoteOut(Note):
    """A Note as returned over HTTP — adds the `filename` the client keys URLs
    on (always `<note_id>.json`)."""

    filename: str = ""


#############################################################################
#                                                                           #
#                    MODELS FOR INTERACTIVE USER LEARNING                   #
#                                                                           #
#############################################################################


class Review(BaseModel):
    misconception: str


class CreateStudyBubble(BaseModel):
    name: str
    user_id: str
    description: str = ""
    domains: List[str] = Field(default_factory=list)
    user_goals: List[str] = Field(default_factory=list)


class StudyBubble(CreateStudyBubble):
    id: str
    created_at: datetime


class CreateResearchProject(BaseModel):
    name: str
    description: str = ""
    domains: List[str] = Field(default_factory=list)
    user_goals: List[str] = Field(default_factory=list)


class ResearchProject(CreateResearchProject):
    id: str
    created_at: datetime


#############################################################################
#                                                                           #
#                            LEARNING MODELS                                #
#                                                                           #
#############################################################################


#############################################################################
#                                                                           #
#                    ARCHIVEING AND CUNKING MODELS                          #
#                                                                           #
#############################################################################


# ---------------------------ARCHIVE MODELS---------------------------
class _AnalysedChunks(BaseModel):
    chunk_fingerprint: str
    chunk_index: str
    analysis: str


# not in use
class CachedChunkAnalysis(BaseModel):
    bubble_id: str
    note_id: str
    analyses: List[_AnalysedChunks]


class ArchivedNoteContent(BaseModel):
    version: float
    content: str


class ArchivedNote(BaseModel):
    note_id: str
    note_name: str
    versions: List[ArchivedNoteContent]


# ---------------------------CACHE MODELS---------------------------
class NoteQueryToCache(BaseModel):
    note_id: str
    bubble_id: str
    semantic_version: float
    content: TranslatedQuery


class RetrievedDocsCache(BaseModel):
    domain: str
    content: list[Document]
    semantic_fingerprint: str

    # metadata
    note_id: str
    bubble_id: str


class AnalysisToCache(BaseModel):
    analysis: str

    # metadata
    note_id: str
    bubble_id: str


# not in use
class SemanticFingerprint(BaseModel):
    note_id: str
    bubble_id: str
    semantic_version: float

    def canonical(self) -> str:
        payload = {
            "note_id": self.note_id,
            "bubble_id": self.bubble_id,
            "semantic_version": self.semantic_version,
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True)

    def hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()
