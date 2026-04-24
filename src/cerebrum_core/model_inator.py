import hashlib
import json
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from pydantic import BaseModel, Field


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


class OllamaConfig(BaseModel):
    url: str = ""


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


class FileMetadata(BaseModel):
    title: str
    domain: str
    subject: str
    authors: str | List[str]
    keywords: str | List[str]


class Chunk(BaseModel):
    pass


#############################################################################
#                                                                           #
#                          MODELS FOR NOTES                                 #
#                                                                           #
#############################################################################


class NoteContent(BaseModel):
    # This is exactly the "document" wrapper AppFlowy expects
    document: Dict[str, Any]  # The full JSON of the document


class InkStroke(BaseModel):
    id: str
    points: List[Dict[str, float]]
    color: str
    thickness: float


class NoteBase(BaseModel):
    title: str
    note_id: str = ""
    bubble_id: str = ""
    # AppFlowy expects a "document" key here
    content: NoteContent
    ink: List[InkStroke] = Field(default_factory=list)


class NoteMetadata(BaseModel):
    content_hash: str = ""
    content_version: float = 0
    ink_hash: str = ""
    ink_version: float = 0
    last_modified: datetime = Field(default_factory=lambda: datetime.now())


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


class NoteHistory(BaseModel):
    content: List[ContentDiff] = Field(default_factory=list)
    ink: List[InkDiff] = Field(default_factory=list)


class NoteStorage(NoteBase):
    metadata: NoteMetadata = Field(default_factory=NoteMetadata)
    history: NoteHistory = Field(default_factory=NoteHistory)


class NoteOut(NoteBase):
    filename: str


#############################################################################
#                                                                           #
#                    MODELS FOR INTERACTIVE USER LEARNING                   #
#                                                                           #
#############################################################################


class Review(BaseModel):
    misconception: str


class CreateStudyBubble(BaseModel):
    name: str
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
