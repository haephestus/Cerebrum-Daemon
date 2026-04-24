import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import routes_bubble, routes_knowledgebase, routes_learning_center, routes_user
from cerebrum_core.user_inator import ConfigManager
from cerebrum_core.utils.file_util_inator import CerebrumPaths
from cerebrum_core.utils.registry.file_chunk_registry_inator import (
    FileChunkRegisterInator,
)
from cerebrum_core.utils.registry.file_registry_inator import FileRegisterInator
from cerebrum_core.utils.registry.note_chunk_registry_inator import (
    NoteChunkRegisterInator,
)
from cerebrum_core.utils.registry.note_registry_inator import NoteRegisterInator

config_manager = ConfigManager()
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cerebrum_paths = CerebrumPaths()
    cerebrum_paths.init_cerebrum_dirs()
    # SQL DBs necessary for file processing
    app.state.file_registry = FileRegisterInator()
    app.state.note_registry = NoteRegisterInator()
    app.state.file_chunk_registry = FileChunkRegisterInator()
    app.state.note_chunk_registry = NoteChunkRegisterInator()

    # ROUTES for api level control
    app.include_router(routes_user.configs_router)
    app.include_router(routes_knowledgebase.router)
    app.include_router(routes_bubble.bubble_router)
    # app.include_router(routes_projects.project_router)
    app.include_router(routes_learning_center.router_learn)

    yield


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
    uvicorn.run(app, host="0.0.0.0", port=8000)
