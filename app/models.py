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


class EpisodePatch(BaseModel):
    """Partial in-place edit of an existing episode. Only fields explicitly set
    in the request are changed. `episode_id` is the key and cannot be patched
    (rename = delete + re-POST). Setting `summary` triggers a re-embed, since
    vector search runs over the summary's embedding."""
    instance: Optional[str] = None
    project: Optional[str] = None
    summary: Optional[str] = None
    raw_exchange: Optional[str] = None
    tags: Optional[list[str]] = None


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
    include_superseded: bool = Field(
        default=False,
        description="If True, include episodes that have been superseded by consolidation. "
                    "Default False: search returns only canonical / unsuperseded episodes."
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
    # ─── Consolidation lineage (per nightly_episode_consolidation plan) ───
    # superseded_by: this episode is a stale snapshot; points at the canonical
    # successor. Default-filtered out of vector_search; retrievable for audit.
    superseded_by: Optional[str] = None
    superseded_at: Optional[datetime] = None
    # consolidated_from: this episode IS the canonical merge; lists the
    # original episode_ids that fed into it. Lineage is explicit + reversible.
    consolidated_from: list[str] = []

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
    boosted_score: Optional[float] = Field(
        None,
        description="Priority-boosted score used for ranking (similarity × multiplier)"
    )
    priority_multiplier: Optional[float] = Field(
        None,
        description="Score multiplier applied (1.5 for critical/correction, 1.0 otherwise)"
    )


class DashboardStats(BaseModel):
    """Aggregated statistics for the dashboard."""
    total_episodes: int = 0
    episodes_by_project: dict[str, int] = {}
    episodes_by_instance: dict[str, int] = {}
    most_retrieved_episodes: list[EpisodeOut] = []
    recent_episodes: list[EpisodeOut] = []
    total_retrievals: int = 0
