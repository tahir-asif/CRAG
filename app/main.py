from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Codebase RAG API", version="0.1.0")
# Currently in development so CORS allows everything
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_STUB_STORE: dict[str, list[RetrievedChunk]] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/repos")
def list_repos():
    return {"repos": list(_STUB_STORE.keys())}
