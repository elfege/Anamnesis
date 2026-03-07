from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


# ─── Request Models ──────────────────────────────────────────────

class EpisodeCreate(BaseModel):
    """Payload for ingesting a new episode."""
    episode_id: str = Field(                                       # e.g. "ep_20260226_migration_lesson"
        ...,
        description="Unique episode identifier (format: ep_YYYYMMDD_description)"
    )
    instance: str = Field(                                         # e.g. "office-nvr"
        ...,
        description="Claude instance that produced this episode"
    )
    project: str = Field(                                          # e.g. "0_MOBIUS.NVR"
        ...,
        description="Project context for the episode"
    )
    summary: str = Field(                                          # the text that gets embedded
        ...,
        description="Distilled lesson or experience (this gets embedded)"
    )
    raw_exchange: Optional[str] = Field(                           # original conversation excerpt
        None,
        description="Original conversation excerpt for high-fidelity reconstruction"
    )
    tags: list[str] = Field(                                       # e.g. ["failure", "debugging"]
        default_factory=list,
        description="Categorical tags for the episode"
    )


class EpisodeSearchRequest(BaseModel):
    """Payload for vector similarity search."""
    query_text: str = Field(
        ...,
        description="Text to search by (will be embedded and compared)"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    project_filter: Optional[str] = Field(
        None,
        description="Filter results to a specific project"
    )
    instance_filter: Optional[str] = Field(
        None,
        description="Filter results to a specific instance"
    )
    tag_filter: Optional[list[str]] = Field(
        None,
        description="Filter results to episodes containing these tags"
    )


# ─── Response Models ─────────────────────────────────────────────

class EpisodeOut(BaseModel):
    """Full episode as returned by the API."""
    episode_id: str
    instance: str
    project: str
    summary: str
    raw_exchange: Optional[str] = None
    tags: list[str] = []
    timestamp: datetime
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EpisodeSearchResult(EpisodeOut):
    """Episode with similarity score from vector search."""
    similarity_score: float = Field(
        ...,
        description="Cosine similarity score (0.0 to 1.0)"
    )


class DashboardStats(BaseModel):
    """Aggregated statistics for the dashboard."""
    total_episodes: int = 0
    episodes_by_project: dict[str, int] = {}
    episodes_by_instance: dict[str, int] = {}
    most_retrieved_episodes: list[EpisodeOut] = []
    recent_episodes: list[EpisodeOut] = []
    total_retrievals: int = 0
