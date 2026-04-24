# pass chunk coordinates or bytes to the class
# fetch the chunks from markdown artifacts
from cerebrum_core.utils.registry.note_chunk_registry_inator import \
    NoteChunkRegisterInator


class ChunkAnalyserInator:
    def __init__(self, note_id: str, note_chunks) -> None:
        self.note_id = note_id
        self.note_chunks = note_chunks
        self.note_chunk_registry = NoteChunkRegisterInator()

    # retrieve chunk coordinates from registry
    # fetch chunk content from md artifacts
    # compare chunk_fingerprint between current note and registry data note version
    # if chunk_fingerprint changed
    #     pass chunk to translated query (analyse chunk)
    #     analyse chunk (store retrieved docs in bubble cache)
    #     store chunk analysis
    #     provide meta analysis for the whole file
    def chunk_analyser_inator(self, prompt: str, top_k_chunks: int = 3) -> str:
        pass

    # uses chunk coordinates from registry to get chun content
    # from the markdown file
    def chunk_fetcher_inator(
        self,
        cached_note_path: str,
        chunk_index: str,
        chunk_start,
        chunk_end,
    ):
        self.note_chunk_registry.
