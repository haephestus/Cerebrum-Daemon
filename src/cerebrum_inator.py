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
    routes_study_plan,
    routes_test,
    routes_user,
)
from api.auth.daemon_auth_inator import get_or_create_daemon_key
from cerebrum_core.engrams.storage.sqlite_repository import NoteEngramRepository
from cerebrum_core.learning_center_inator import run_generation_queue_worker
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.chunking_queue_inator import file_processing_queue
from cerebrum_core.utils.database.file_chunk_registry_inator import (
    FileChunkRegisterInator,
)
from cerebrum_core.utils.database.file_registry_inator import FileRegisterInator
from cerebrum_core.utils.database.note_chunk_registry_inator import (
    NoteChunkRegisterInator,
)
from cerebrum_core.utils.database.study_plan import StudyPlanRegisterInator
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.ollama_compat.ollama_parser_inator import (
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
    # NON-BLOCKING OFFLINE MANIFEST GENERATION (Type-Safe Native)
    # ─────────────────────────────────────────────────────────────
    # Safely await the worker on a native asyncio background thread.
    # This completely satisfies Pyright and bypasses strict module checks.
    await asyncio.to_thread(_sync_manifest_worker)

    # SQL DBs necessary for file processing
    app.state.file_registry = FileRegisterInator()
    app.state.note_registry = NoteEngramRepository()
    app.state.file_chunk_registry = FileChunkRegisterInator()
    app.state.note_chunk_registry = NoteChunkRegisterInator()
    app.state.study_plan_registry = StudyPlanRegisterInator()

    # ROUTES for api level control
    app.include_router(routes_user.user_router)
    app.include_router(routes_knowledgebase.router)
    app.include_router(routes_bubble.bubble_router)
    # app.include_router(routes_projects.project_router)
    app.include_router(routes_learning_center.router_learn)
    app.include_router(routes_test.router_test)
    app.include_router(routes_study_plan.router_study_plan)

    file_processing_queue.start()
    queue_task = asyncio.create_task(
        run_generation_queue_worker(app.state.note_registry)
    )
    yield

    queue_task.cancel()


def create_api_server():
    """
    Initializes server config and middleware.
    """

    # %%
    app = FastAPI(lifespan=lifespan)

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
