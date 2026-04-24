import json
import logging

from agents.rose import RosePrompts
from cerebrum_core.model_inator import NoteStorage
from cerebrum_core.utils.analyser_inator import NoteAnalyserInator
from cerebrum_core.utils.cache_inator import AnalysisCacheInator
from cerebrum_core.utils.file_util_inator import CerebrumPaths

logger = logging.getLogger(__name__)


def generate_engram():
    # WARN: PURPOSE: progress tracking (based of learning goals)
    #                note analysis based of current and historic notes
    #                engram(quizzes, mockexams) generation

    # TODO: implement models
    '''
    # Use model matching to generate quizzes, or analysis or mockexams
    def assesment_maker(raw: str, mode: str):
        """
        mode: quiz. analysis, mockexams
        """
        match mode:
            case "quiz":
                # parse string into a QuizModel
                return QuizModel.parse_from_string(raw)
            case "analysis":
                return AnalysisModel.parse_from_string(raw)
            case  "mock_exam":
                return MockExamModel.parse_from_string(raw)
            case _:
            # just return raw string
                return raw
    '''
    pass


# TODO: generate engram(readings, quizzes, mock exams, flash cards)
#       run adapative spaced repetition
#       place quizzes in bubble specific folders
#       store historic engrams in bubble specific folders

# cerebrum_core/learning_center_inator.py


def passive_analysis(
    note: NoteStorage, prompt: str, cache_manager: AnalysisCacheInator
) -> str:
    """
    Perform passive analysis on a note and cache the result.

    This function:
    1. Chunks the note
    2. Archives it (via NoteAnalyserInator)
    3. Runs analysis against knowledge base
    4. Caches the result

    Args:
        note: The note to analyze
        prompt: Analysis prompt template
        cache_manager: Cache manager for storing results

    Returns:
        Analysis result string
    """
    try:
        logger.info(
            f"Starting passive analysis for note {note.note_id} v{note.metadata.content_version}"
        )

        # Initialize analyzer (this chunks and archives the note)
        analyzer = NoteAnalyserInator(note=note, generate_artifact=True)

        logger.info(
            f"Initialized analyzer: {len(analyzer.chunks)} chunks, "
            f"{len(analyzer.translation_results)} queries"
        )

        # Run analysis (retrieves from KB and generates response)
        analysis_result = analyzer.analyser_inator(prompt=prompt, top_k_chunks=5)
        if not analysis_result:
            return "no analysis for this note"

        logger.info(
            f"Completed analysis for note {note.note_id} v{note.metadata.content_version}"
        )

        # Cache the result with metadata
        cache_manager.cache_analysis(
            content_version=note.metadata.content_version,
            analysis=analysis_result,
            metadata={
                "chunks_count": len(analyzer.chunks),
                "queries_count": len(analyzer.translation_results),
                "retrieved_docs": len(analyzer.retrieved_docs),
                "note_title": note.title,
            },
        )

        logger.info(f"Cached analysis for note {note.note_id}")

        return analysis_result

    except Exception as e:
        logger.error(f"Failed to analyze note {note.note_id}: {e}", exc_info=True)
        raise


def active_analysis(bubble_id: str, filename: str):

    stored_note = CerebrumPaths().note_path(bubble_id=bubble_id, filename=filename)
    note = NoteStorage(**json.loads(stored_note.read_text(encoding="utf-8")))
    cache_manager = AnalysisCacheInator(bubble_id, filename)

    prompt = RosePrompts.get_prompt("rose_note_analyser")
    if not prompt:
        return "Prompt can not be none"

    try:
        logger.info(
            f"Starting active analysis for note {note.note_id} v{note.metadata.content_version}"
        )

        # Initialize analyzer (this chunks and archives the note)
        analyzer = NoteAnalyserInator(note=note, generate_artifact=True)

        logger.info(
            f"Initialized analyzer: {len(analyzer.chunks)} chunks, "
            f"{len(analyzer.translation_results)} queries"
        )

        # Run analysis (retrieves from KB and generates response)
        analysis_result = analyzer.analyser_inator(prompt=prompt, top_k_chunks=5)
        if not analysis_result:
            logger.info("No analysis for this note")

        logger.info(
            f"Completed analysis for note {note.note_id} v{note.metadata.content_version}"
        )

        # Cache the result with metadata
        cache_manager.cache_analysis(
            content_version=note.metadata.content_version,
            analysis=analysis_result,
            metadata={
                "chunks_count": len(analyzer.chunks),
                "queries_count": len(analyzer.translation_results),
                "retrieved_docs": len(analyzer.retrieved_docs),
                "note_title": note.title,
            },
        )

        logger.info(f"Cached analysis for note {note.note_id}")

        return analysis_result

    except Exception as e:
        logger.error(f"Failed to analyze note {note.note_id}: {e}", exc_info=True)
        raise
