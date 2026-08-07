import asyncio  # Swapped to native asyncio for foolproof Pyright typing
import json
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import (
    routes_bubble,
    routes_knowledgebase,
    routes_learning_center,
    routes_org,
    routes_study_plan,
    routes_suggested_reading,
    routes_test,
    routes_user,
)
from api.auth.daemon_auth_inator import get_or_create_daemon_key
from common.deploy_config_inator import is_local
from api.middleware.daemon_auth_middleware import DaemonAuthMiddleware
from database.file_chunk_registry_inator import FileChunkRegisterInator
from database.file_registry_inator import FileRegisterInator
from database.note_chunk_registry_inator import NoteChunkRegisterInator
from database.note_engram_repository import NoteEngramRepository
from database.planner import StudyPlanRegisterInator
from cerebrum_core.engrams.grading.worker import SQLiteWorkerLoop, run_worker
from cerebrum_core.learning_center_inator import run_generation_queue_worker
from cerebrum_core.user_inator import ConfigManager
from notes.chunking_queue_inator import file_processing_queue
from common.file_util_inator import CerebrumPaths
from common.ollama_compat.ollama_parser_inator import (
    OllamaManifestGenerator,
)

config_manager = ConfigManager()
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)


def _sync_manifest_worker():
    """
    Synchronous worker executing standard operations.
    Isolated here to be cleanly called inside the async thread pool.
    """
    manifest_path = CerebrumPaths().config_root_dir() / "models_manifest.json"

    if not manifest_path.exists():
        print("'models_manifest.json' not found! Initializing scraping routine...")
        try:
            engine = OllamaManifestGenerator()

            # Executing your synchronous catalog builder without 'await' conflict
            manifest_data = engine.build_master_manifest()

            # Ensure directories exist and dump data using standard library I/O
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=4)

            print(f" Success! Master manifest baked to disk at: {manifest_path}")
        except Exception as e:
            print(f" Failed to auto-generate models manifest on daemon boot: {e}")
    else:
        print(" Master models manifest confirmed on disk. Skipping scrape.")


# TODO: async file processing: schedule md_conversion and md_chunking
# so files are ready for analysis
@asynccontextmanager
async def lifespan(app: FastAPI):
    cerebrum_paths = CerebrumPaths()
    cerebrum_paths.init_cerebrum_dirs()

    # ─────────────────────────────────────────────────────────────
    # OFFLINE MANIFEST GENERATION — deferred, not awaited
    # ─────────────────────────────────────────────────────────────
    # This scrapes/bakes the models manifest and can be slow (network I/O).
    # Awaiting it here would block the server from accepting traffic and can
    # blow a serverless cold-boot budget (Leapcell ~10s), so we fire it as a
    # background task and let the server come up immediately.
    manifest_task = asyncio.create_task(asyncio.to_thread(_sync_manifest_worker))

    # SQL DBs necessary for file processing
    app.state.file_registry = FileRegisterInator()
    app.state.note_registry = NoteEngramRepository()
    app.state.file_chunk_registry = FileChunkRegisterInator()
    app.state.note_chunk_registry = NoteChunkRegisterInator()
    app.state.study_plan_registry = StudyPlanRegisterInator()

    # ROUTES for api level control
    app.include_router(routes_user.user_router)
    app.include_router(routes_org.org_router)
    app.include_router(routes_knowledgebase.router)
    app.include_router(routes_bubble.bubble_router)
    # app.include_router(routes_projects.project_router)
    app.include_router(routes_learning_center.router_learn)
    app.include_router(routes_test.router_test)
    app.include_router(routes_study_plan.router_study_plan)
    app.include_router(routes_suggested_reading.router_suggested_reading)

    # In-process background workers only run in local mode. On serverless
    # (cloud) the process may be frozen/killed between requests, so these loops
    # would silently stall — there they belong in a separate always-on worker
    # service. Gating them here also keeps cold-boot within Leapcell's ~10s
    # budget, since we don't spin up drain loops on the request path.
    queue_task = grading_task = None
    if is_local():
        file_processing_queue.start()
        queue_task = asyncio.create_task(
            run_generation_queue_worker(app.state.note_registry)
        )
        # Grading worker: drains grading_jobs (long_question + short_question).
        # vector_store/embedder are None — note-grading context comes from the
        # concrete cached->KB retriever (grading.context_retrieval), and the
        # answer-embedding/regression path is gated off until a concrete answer
        # vector store is supplied. use_cloud=None defers to should_use_cloud().
        grading_task = asyncio.create_task(
            run_worker(
                app.state.note_registry,
                vector_store=None,  # type: ignore[arg-type]
                embedder=None,  # type: ignore[arg-type]
                loop=SQLiteWorkerLoop(app.state.note_registry),
            )
        )
    yield

    manifest_task.cancel()
    if queue_task:
        queue_task.cancel()
    if grading_task:
        grading_task.cancel()


def create_api_server():
    """
    Initializes server config and middleware.
    """

    # %%
    app = FastAPI(lifespan=lifespan)

    # Order matters: middleware added LAST runs OUTERMOST. CORS is added after
    # DaemonAuth so it wraps it — that way browser preflight (OPTIONS) is
    # answered by CORS before the key check, and the 401 from a bad/missing
    # key still carries CORS headers instead of tripping a Flutter-web client.
    app.add_middleware(DaemonAuthMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"message": "Cerebrum API is running"}

    # include routers
    # app.include_router(chat.router)
    return app


app = create_api_server()

if __name__ == "__main__":
    # Important so uvicorn doesn't run on import
    key = get_or_create_daemon_key()
    print(f"\n{'='*60}\nDaemon key (enter this in the app once): {key}\n{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
