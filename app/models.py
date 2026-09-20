from pydantic import BaseModel, ConfigDict, Field


class IngestRequest(BaseModel):
    repo_url: str = Field(..., description="Public GitHub repo URL")
    branch: str | None = Field(
        None,
        description="Branch to clone. If omitted, uses the repo's default branch.",
    )
    file_extensions: list[str] = Field(
        default=[".py"], description="File extensions to index"
    )
    path_filter: str | None = Field(
        None, description="Optional subdirectory, e.g. 'src/'"
    )


class IngestResponse(BaseModel):
    repo_name: str
    files_indexed: int
    chunks_created: int
    status: str


class Chunk(BaseModel):
    content: str
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str = Field(..., description="function | class | module")
    name: str | None = None


class RetrievedChunk(Chunk):
    score: float
    source: str = Field(..., description="vector | bm25 | hybrid | rerank | stub")


class LLMResponse(BaseModel):
    answer: str
    cited_indices: list[int] = Field(default_factory=list)


class Citation(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    name: str | None = None


class QueryRequest(BaseModel):
    question: str
    repo_name: str | None = None
    top_k: int = 10
    rerank_top_k: int = 5
    api_key: str | None = Field(
        None, description="Optional Groq API key. Falls back to server default."
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


class ChunkMetadata(BaseModel):
    """Validates the metadata shape for chunks."""

    model_config = ConfigDict(extra="ignore")

    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str = ""
